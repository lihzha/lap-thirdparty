import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from openpi.models.gemma import get_config as get_gemma_config
import openpi.models.model as _model
from openpi.models.model import Observation
import openpi.models.pi0 as _pi0
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
from typing_extensions import override

from openpi_cot.models.adapters.gemma_adapter import Gemma2ModuleWithDecode
from openpi_cot.models.adapters.gemma_adapter import Gemma3ModuleWithDecode
from openpi_cot.models.adapters.gemma_adapter import ModuleWithDecode
from openpi_cot.models.adapters.model_adapter import CoTObservation
from openpi_cot.models.adapters.model_adapter import preprocess_observation
from openpi_cot.models.gemma2 import get_config as get_gemma2_config
from openpi_cot.models.gemma3 import get_config as get_gemma3_config
import openpi_cot.models.pi_cot_config as _pi_cot_config

logger = logging.getLogger("openpi")


def cross_entropy_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    mask: jnp.ndarray | None = None,
    axis: int = -1,
    train: bool = True,
    *,
    per_example: bool = False,
) -> jnp.ndarray:
    """
    Args
    ----
      logits : (..., V)   – raw scores.
      labels : (...)      – int32 / int64 class‑ids, same leading shape as logits without the class dim.
      mask   : (...) or None – 0/1 or bool; broadcastable to `labels`.
      axis   : int        – class dimension in `logits`.
      train  : bool       – if True → mean loss, else → summed loss.

    Returns
    -------
      If per_example=False (default): scalar mean (train=True) or scalar sum (train=False).
      If per_example=True: per-example mean over non-batch dims (shape [B]).
    """
    # log‑probs
    log_probs = nnx.log_softmax(logits, axis=axis)  # (..., V)

    # gather log‑prob of the gold class
    gather_idx = jnp.expand_dims(labels.astype(jnp.int32), axis=axis)  # (..., 1)
    gold_logp = jnp.take_along_axis(log_probs, gather_idx, axis=axis)  # (..., 1)
    loss = -gold_logp.squeeze(axis)  # (...)

    # optional masking
    if per_example:
        # Reduce over all non-batch dims (assume batch is leading dimension)
        reduce_axes = tuple(range(1, loss.ndim))
        if mask is not None:
            loss = loss * mask
            denom = jnp.maximum(mask.sum(axis=reduce_axes), 1)  # [B]
        else:
            # Mean over all trailing dims
            denom = jnp.prod(jnp.array(loss.shape[1:]))
        total = loss.sum(axis=reduce_axes)
        return total / denom
    if mask is not None:
        loss = loss * mask
        denom = jnp.maximum(mask.sum(), 1)  # avoid ÷0 for empty mask
    else:
        denom = loss.size
    total = loss.sum()
    return total / denom if train else total


class PiCoT(_pi0.Pi0):
    EOS_ID = 1  # TODO: hard-coded for PaliGemma

    def __init__(self, config: _pi_cot_config.PiCoTConfig, rngs: nnx.Rngs):
        _model.BaseModel.__init__(self, config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.aug_wrist_image = config.aug_wrist_image
        self.image_keys = config.image_keys
        # Loss/control knobs
        self.enable_action_training = bool(getattr(config, "enable_action_training", False))
        self.enable_langact_training = bool(getattr(config, "enable_langact_training", True))
        self.enable_prediction_training = bool(getattr(config, "enable_prediction_training", False))
        self.language_loss_weight = float(getattr(config, "language_loss_weight", 1.0))
        self.action_loss_weight = float(getattr(config, "action_loss_weight", 1.0))
        self.prediction_loss_weight = float(getattr(config, "prediction_loss_weight", 1.0))
        # Backward compatibility flag used in a few places
        self.lang_action_only = not self.enable_action_training
        if "gemma2" in config.paligemma_variant:
            assert "gemma2" in config.action_expert_variant, "gemma2 must be used for both LLM and action expert"
            paligemma_config = get_gemma2_config(config.paligemma_variant)
            action_expert_config = get_gemma2_config(config.action_expert_variant)
            module = Gemma2ModuleWithDecode
        elif "gemma3" in config.paligemma_variant:
            assert "gemma3" in config.action_expert_variant, "gemma3 must be used for both LLM and action expert"
            paligemma_config = get_gemma3_config(config.paligemma_variant)
            action_expert_config = get_gemma3_config(config.action_expert_variant)
            module = Gemma3ModuleWithDecode
        else:
            paligemma_config = get_gemma_config(config.paligemma_variant)
            action_expert_config = get_gemma_config(config.action_expert_variant)
            module = ModuleWithDecode

        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        # Conditionally set the positional embedding type for the image model
        if "gemma3" in config.paligemma_variant:
            # For Gemma3, use sinusoidal positional embeddings to avoid size mismatch
            img = nnx_bridge.ToNNX(
                _siglip.Module(
                    num_classes=paligemma_config.width,
                    variant="So400m/14",
                    pool_type="none",
                    scan=True,
                    dtype_mm=config.dtype,
                    posemb="sincos2d", # needed for size-mismatch
                )
            )
        else:
            # For other models, use the original default (learnable embeddings)
            img = nnx_bridge.ToNNX(
                _siglip.Module(
                    num_classes=paligemma_config.width,
                    variant="So400m/14",
                    pool_type="none",
                    scan=True,
                    dtype_mm=config.dtype,
                )
            )

        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    def _embed_images(
        self, obs: CoTObservation | Observation, num_frames: int | None = None
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, "b s"],
    ]:
        """Embed images, optionally limiting to first num_frames.

        Args:
            obs: Observation containing images with shape [b, t, h, w, c]
            num_frames: If specified, only use first num_frames. If None, use all frames.

        Returns:
            (image_tokens, image_mask, image_ar_mask)
        """
        input_mask = []
        tokens = []
        _img_ar_masks = []

        for name in obs.images:
            image = obs.images[name]
            b, t, h, w, c = image.shape

            # Limit to num_frames if specified
            if num_frames is not None:
                image = image[:, :num_frames]
                t = num_frames

            # Flatten: [b*t, h, w, c]
            image_flat = image.reshape(b * t, h, w, c)
            image_tokens, _ = self.PaliGemma.img(image_flat, train=False)
            # image_tokens: [b*t, num_patches, d]

            num_patches = image_tokens.shape[1]
            # Reshape: [b, t*num_patches, d]
            image_tokens = image_tokens.reshape(b, t * num_patches, -1)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=t * num_patches,
                )
            )
            # All image tokens attend to each other
            _img_ar_masks += [False] * (t * num_patches)

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        img_ar_mask = jnp.array(_img_ar_masks)
        img_ar_mask = einops.repeat(img_ar_mask, "s -> b s", b=tokens.shape[0])

        return tokens, input_mask, img_ar_mask

    def _embed_text(
        self, tokenized_text, text_mask, text_ar_mask
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, "b s"],
    ]:
        """Embed tokenized text."""
        text_tokens = self.PaliGemma.llm(tokenized_text, method="embed")
        return text_tokens, text_mask, text_ar_mask

    @at.typecheck
    def embed_prefix(
        self, obs: CoTObservation | Observation, num_frames: int | None = None
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, "b s"],
    ]:
        """Embed images and text for the prefix.

        Args:
            obs: Observation containing images and tokenized text
            num_frames: If specified, only use first num_frames of images. If None, use all frames.

        Returns:
            (prefix_tokens, prefix_mask, prefix_ar_mask)
        """
        img_tokens, img_mask, img_ar_mask = self._embed_images(obs, num_frames)

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            text_tokens, text_mask, text_ar_mask = self._embed_text(
                obs.tokenized_prompt,
                obs.tokenized_prompt_mask,
                obs.tokenized_langact_mask,
            )

            if text_ar_mask is not None:
                ar_mask = jnp.concatenate([img_ar_mask, text_ar_mask], axis=1)
            else:
                text_ar_mask = jnp.array([False] * text_mask.shape[1])
                text_ar_mask = einops.repeat(text_ar_mask, "s -> b s", b=img_ar_mask.shape[0])
                ar_mask = jnp.concatenate([img_ar_mask, text_ar_mask], axis=1)
            tokens = jnp.concatenate([img_tokens, text_tokens], axis=1)
            input_mask = jnp.concatenate([img_mask, text_mask], axis=1)
        else:
            tokens, input_mask, ar_mask = img_tokens, img_mask, img_ar_mask

        return tokens, input_mask, ar_mask

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: CoTObservation | Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, "*b ah"], at.Float[at.Array, ""], at.Float[at.Array, ""], dict[str, at.Array]]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        # Assume reasoning is already tokenized for compute_loss. For inference, we tokenize on-the-fly.
        observation = preprocess_observation(
            preprocess_rng, observation, train=train, image_keys=self.image_keys, aug_wrist_image=self.aug_wrist_image
        )

        # Optimization: if prediction training is enabled, encode all frames once and reuse
        if self.enable_prediction_training and observation.tokenized_prediction is not None:
            # Encode all frames once
            img_tokens_all, img_mask_all, img_ar_mask_all = self._embed_images(observation, num_frames=None)

            # Calculate num_patches: total tokens divided by total frames across all images
            total_frames = sum(img.shape[1] for img in observation.images.values())
            num_patches = img_tokens_all.shape[1] // total_frames

            # Extract first frame tokens for each image
            first_frame_tokens = []
            first_frame_masks = []
            first_frame_ar_masks = []

            offset = 0
            for name in observation.images:
                num_frames_in_image = observation.images[name].shape[1]

                # Extract first frame for this image
                first_frame_tokens.append(img_tokens_all[:, offset : offset + num_patches])
                first_frame_masks.append(img_mask_all[:, offset : offset + num_patches])
                first_frame_ar_masks.append(img_ar_mask_all[:, offset : offset + num_patches])

                # Move offset by all frames of this image
                offset += num_frames_in_image * num_patches

            img_tokens_first = jnp.concatenate(first_frame_tokens, axis=1)
            img_mask_first = jnp.concatenate(first_frame_masks, axis=1)
            img_ar_mask_first = jnp.concatenate(first_frame_ar_masks, axis=1)
        else:
            # Only encode first frame
            img_tokens_first, img_mask_first, img_ar_mask_first = self._embed_images(observation, num_frames=1)

        # Build prefix for langact/action losses (first frame + regular text)
        text_tokens, text_mask, text_ar_mask = self._embed_text(
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.tokenized_langact_mask,
        )
        prefix_tokens = jnp.concatenate([img_tokens_first, text_tokens], axis=1)
        prefix_mask = jnp.concatenate([img_mask_first, text_mask], axis=1)
        if text_ar_mask is not None:
            prefix_ar_mask = jnp.concatenate([img_ar_mask_first, text_ar_mask], axis=1)
        else:
            text_ar_mask = jnp.array([False] * text_mask.shape[1])
            text_ar_mask = einops.repeat(text_ar_mask, "s -> b s", b=prefix_mask.shape[0])
            prefix_ar_mask = jnp.concatenate([img_ar_mask_first, text_ar_mask], axis=1)

        total_loss = 0.0
        token_accuracy = jnp.array(0.0)
        critical_token_accuracy = jnp.array(0.0)

        # Additional metrics to return
        metrics = {}

        # Cross-entropy (language/reasoning) loss
        if self.enable_langact_training:
            attn_mask_lang = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
            positions_lang = jnp.cumsum(prefix_mask, axis=1) - 1
            (prefix_out, _), _ = self.PaliGemma.llm(
                [prefix_tokens, None], mask=attn_mask_lang, positions=positions_lang
            )

            # Predict next tokens over the reasoning span
            shift_labels = observation.tokenized_prompt[:, 1:]
            max_len = observation.tokenized_langact_mask.shape[1]
            shift_tokens = prefix_out[:, -max_len:-1, :]
            shift_logits = self.PaliGemma.llm(shift_tokens, method="decode")

            langact_and_pad_mask = jnp.logical_and(
                observation.tokenized_langact_mask[:, 1:],
                observation.tokenized_prompt_mask[:, 1:],
            )

            ex_mask = jnp.asarray(observation.sample_mask)[..., None]
            token_mask = langact_and_pad_mask * ex_mask

            lang_loss = cross_entropy_loss(
                shift_logits,
                shift_labels,
                mask=token_mask,
                axis=-1,
                train=True,
                per_example=True,
            )
            metrics["lang_loss"] = lang_loss
            total_loss = total_loss + self.language_loss_weight * lang_loss

            # Compute token accuracy
            predictions = jnp.argmax(shift_logits, axis=-1)
            correct = (predictions == shift_labels).astype(jnp.float32)
            masked_correct = correct * token_mask
            num_tokens = jnp.maximum(token_mask.sum(), 1.0)
            token_accuracy = masked_correct.sum() / num_tokens

            # Compute critical token accuracy
            critical_token_mask = observation.crictical_token_mask[:, 1:] * ex_mask
            critical_correct = correct * critical_token_mask
            num_critical_tokens = jnp.maximum(critical_token_mask.sum(), 1.0)
            critical_token_accuracy = critical_correct.sum() / num_critical_tokens

        # Prediction (cross-entropy) loss - independent of langact loss
        if self.enable_prediction_training and observation.tokenized_prediction is not None:
            # Reuse already-encoded image tokens (img_tokens_all was computed above)
            # Create separate prefix for prediction: all frames + prediction text
            text_tokens_pred, text_mask_pred, text_ar_mask_pred = self._embed_text(
                observation.tokenized_prediction,
                observation.tokenized_prediction_mask,
                observation.tokenized_prediction_langact_mask,
            )
            prefix_tokens_pred = jnp.concatenate([img_tokens_all, text_tokens_pred], axis=1)
            prefix_mask_pred = jnp.concatenate([img_mask_all, text_mask_pred], axis=1)
            prefix_ar_mask_pred = jnp.concatenate([img_ar_mask_all, text_ar_mask_pred], axis=1)

            # Use prediction-specific prefix
            pred_attn_mask = _pi0.make_attn_mask(prefix_mask_pred, prefix_ar_mask_pred)
            pred_positions = jnp.cumsum(prefix_mask_pred, axis=1) - 1
            (prefix_out_pred, _), _ = self.PaliGemma.llm(
                [prefix_tokens_pred, None], mask=pred_attn_mask, positions=pred_positions
            )

            # Predict next tokens over the prediction langact span
            shift_labels_pred = observation.tokenized_prediction[:, 1:]
            max_len_pred = observation.tokenized_prediction_langact_mask.shape[1]
            shift_tokens_pred = prefix_out_pred[:, -max_len_pred:-1, :]
            shift_logits_pred = self.PaliGemma.llm(shift_tokens_pred, method="decode")

            prediction_and_pad_mask = jnp.logical_and(
                observation.tokenized_prediction_langact_mask[:, 1:],
                observation.tokenized_prediction_mask[:, 1:],
            )

            ex_mask_pred = (
                jnp.asarray(observation.sample_mask)[..., None]
                if observation.sample_mask is not None
                else jnp.ones_like(prediction_and_pad_mask[:, :1])
            )
            token_mask_pred = prediction_and_pad_mask * ex_mask_pred

            pred_loss = cross_entropy_loss(
                shift_logits_pred,
                shift_labels_pred,
                mask=token_mask_pred,
                axis=-1,
                train=True,
                per_example=True,
            )
            metrics["pred_loss"] = pred_loss

            # Compute prediction token accuracy
            predictions_pred = jnp.argmax(shift_logits_pred, axis=-1)
            correct_pred = (predictions_pred == shift_labels_pred).astype(jnp.float32)
            masked_correct_pred = correct_pred * token_mask_pred
            num_tokens_pred = jnp.maximum(token_mask_pred.sum(), 1.0)
            pred_token_accuracy = masked_correct_pred.sum() / num_tokens_pred
            metrics["pred_token_accuracy"] = pred_token_accuracy

            # Compute prediction critical token accuracy if available
            if (
                hasattr(observation, "crictical_prediction_token_mask")
                and observation.crictical_prediction_token_mask is not None
            ):
                critical_pred_mask = observation.crictical_prediction_token_mask[:, 1:] * ex_mask_pred
                critical_correct_pred = correct_pred * critical_pred_mask
                num_critical_pred = jnp.maximum(critical_pred_mask.sum(), 1.0)
                pred_critical_token_accuracy = critical_correct_pred.sum() / num_critical_pred
                metrics["pred_critical_token_accuracy"] = pred_critical_token_accuracy

            total_loss = total_loss + self.prediction_loss_weight * pred_loss

        # Diffusion (actions) loss. TODO: no sample mask for actions! Sample mask may only make sense for langact because it is obtained from cot_policy.is_idle_language_action
        if self.enable_action_training:
            # For action training, text tokens should not use autoregressive masking
            text_ar_mask = jnp.zeros_like(text_mask, dtype=bool)
            prefix_ar_mask = jnp.concatenate([img_ar_mask_first, text_ar_mask], axis=1)

            batch_shape = actions.shape[:-2]
            noise = jax.random.normal(noise_rng, actions.shape)
            time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
            time_expanded = time[..., None, None]
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            u_t = noise - actions
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
            suffix_ar_mask = einops.repeat(suffix_ar_mask, "s -> b s", b=suffix_tokens.shape[0])

            input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
            ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=1)
            attn_mask = _pi0.make_attn_mask(input_mask, ar_mask)
            positions = jnp.cumsum(input_mask, axis=1) - 1

            (_, suffix_out), _ = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            action_loss = jnp.mean(jnp.square(v_t - u_t), axis=(-1, -2))
            metrics["action_loss"] = action_loss
            total_loss = total_loss + self.action_loss_weight * action_loss

        return total_loss, token_accuracy, critical_token_accuracy, metrics

    

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: CoTObservation | Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = preprocess_observation(
            None, observation, train=False, image_keys=self.image_keys, aug_wrist_image=self.aug_wrist_image
        )
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation, num_frames=1)
        prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    ### left padding
    def _sample_reasoning_tokens(self, observation: CoTObservation):
        # ───────────────── 0. Shapes ─────────────────
        observation = preprocess_observation(
            None, observation, train=False, image_keys=self.image_keys, aug_wrist_image=self.aug_wrist_image
        )
        # Inference: only use first frame
        p_tokens, p_mask0, p_ar_mask0 = self.embed_prefix(observation, num_frames=1)  # (B,Tp,D) + (B,Tp)
        b, tp, d = *p_tokens.shape[:2], p_tokens.shape[-1]
        gen_len = observation.tokenized_prompt.shape[1]
        max_len = gen_len + tp

        # For left padding, the prefix occupies the tail window [start, start+tp)
        start = max_len - tp  # <-- NEW

        # ───────────────── 1. Full-length (static-shape) buffers ─────────────────
        # NOTE: we keep the extra +1 column as your "query row" scratch space.
        p_mask = jnp.zeros((b, max_len + 1), dtype=bool)
        p_ar_mask = jnp.zeros((b, max_len + 1), dtype=bool)

        # Place prefix masks into the tail instead of the head
        p_mask = p_mask.at[:, start : start + tp].set(p_mask0)  # <-- CHANGED
        p_ar_mask = p_ar_mask.at[:, start : start + tp].set(p_ar_mask0)  # <-- CHANGED

        # Keep your “query slot” convention
        p_mask = p_mask.at[:, -1].set(1)
        p_ar_mask = p_ar_mask.at[:, -1].set(1)

        # ───────────────── 2. Prefix attention & positions ─────────────────
        # Positions must be contiguous over *real* tokens (ignoring pads).
        # Compute over the full mask, then slice the tail segment used for the prefix call.
        pos_full = jnp.cumsum(p_mask[:, :max_len], axis=1) - 1  # [B, max_len]
        pos_pref = pos_full[:, start : start + tp]  # <-- CHANGED

        # Build an attention mask for just the prefix window
        pref_attn = _pi0.make_attn_mask(
            p_mask[:, start : start + tp],  # <-- CHANGED
            p_ar_mask[:, start : start + tp],
        )  #     (B,Tp,Tp)

        # Forward the prefix at the tail
        (hs, _), kv0 = self.PaliGemma.llm([p_tokens, None], mask=pref_attn, positions=pos_pref)

        curr_h = hs[:, -1:, :]
        curr_id = jnp.argmax(self.PaliGemma.llm(curr_h, method="decode"), axis=-1)  # (B,1)
        curr_h = self.PaliGemma.llm(curr_id, method="embed")
        # Track which sequences have finished (emitted EOS) and keep them finished
        finished = curr_id == self.EOS_ID

        # ───────────────── 3. Static KV cache aligned to tail ─────────────────
        nl, _, _, k, h = kv0[0].shape
        k_cache = jnp.zeros((nl, b, max_len, k, h), dtype=kv0[0].dtype)
        v_cache = jnp.zeros_like(k_cache)

        # Write the prefix keys/values into [start:start+tp]
        k_cache = k_cache.at[:, :, start : start + tp].set(kv0[0])  # <-- CHANGED
        v_cache = v_cache.at[:, :, start : start + tp].set(kv0[1])  # <-- CHANGED

        # ───────────────── 4. Output buffers (unchanged shapes) ─────────────────
        h_buf = jnp.zeros((b, gen_len, d), dtype=hs.dtype).at[:, 0].set(curr_h.squeeze(1))
        id_buf = jnp.zeros((b, gen_len, 1), dtype=jnp.int32).at[:, 0].set(curr_id)
        t0 = 0

        # ───────────────── 5. Body / Cond (only t_abs changes) ─────────────────
        def step(carry):
            (
                curr_h,
                curr_id,
                finished,
                k_cache,
                v_cache,
                p_mask,
                p_ar_mask,
                h_buf,
                id_buf,
                _t,
            ) = carry

            # Sliding window: shift caches and masks left by 1 to free the last slot
            k_cache = jnp.concatenate([k_cache[:, :, 1:], jnp.zeros_like(k_cache[:, :, :1])], axis=2)
            v_cache = jnp.concatenate([v_cache[:, :, 1:], jnp.zeros_like(v_cache[:, :, :1])], axis=2)
            p_mask = jnp.concatenate([p_mask[:, 1:], jnp.zeros_like(p_mask[:, :1])], axis=1)
            p_ar_mask = jnp.concatenate([p_ar_mask[:, 1:], jnp.zeros_like(p_ar_mask[:, :1])], axis=1)

            # Maintain the scratch query column at the end
            p_mask = p_mask.at[:, -1].set(True)
            p_ar_mask = p_ar_mask.at[:, -1].set(True)

            # Build attention for the single query row over the window + scratch
            attn_row = _pi0.make_attn_mask(p_mask, p_ar_mask)[:, -1:, :]  # (B,1,MAX+1)

            # RoPE position for the query: include scratch column in the count
            pos = jnp.sum(p_mask, axis=1, keepdims=True).astype(jnp.int32) - 1

            (next_h, _), kv_new = self.PaliGemma.llm(
                [curr_h, None],
                positions=pos,  # (B,1)
                mask=attn_row,  # (B,1,MAX+1)
                kv_cache=(k_cache, v_cache),
            )

            # Decode → id for next step
            logits = self.PaliGemma.llm(next_h, method="decode")
            next_id_raw = jnp.argmax(logits, axis=-1)
            # Update finished mask and force EOS for finished sequences
            finished = jnp.logical_or(finished, next_id_raw == self.EOS_ID)
            eos_token = jnp.asarray(self.EOS_ID, dtype=next_id_raw.dtype)
            next_id = jnp.where(finished, eos_token, next_id_raw)
            next_h = self.PaliGemma.llm(next_id, method="embed")  # (batch, 1, D)
            # Keep hidden state stable for finished sequences
            next_h = jnp.where(finished[..., None], curr_h, next_h)

            # Write new KV into the last real slot and mark it as real (keep scratch True)
            k_cache = k_cache.at[:, :, -1].set(kv_new[0][:, :, -1])
            v_cache = v_cache.at[:, :, -1].set(kv_new[1][:, :, -1])
            p_mask = p_mask.at[:, -2].set(True)
            p_ar_mask = p_ar_mask.at[:, -2].set(True)

            _t += 1
            h_buf = h_buf.at[:, _t].set(next_h.squeeze(1))
            id_buf = id_buf.at[:, _t].set(next_id)

            return (
                next_h,
                next_id,
                finished,
                k_cache,
                v_cache,
                p_mask,
                p_ar_mask,
                h_buf,
                id_buf,
                _t,
            )

        def cond(carry):
            _, _, finished, *_, t = carry
            unfinished = jnp.any(jnp.logical_not(finished))
            return jnp.logical_and(unfinished, t < gen_len - 1)

        # ───────────────── 5. While-loop ─────────────────

        carry = (
            curr_h,
            curr_id,
            finished,
            k_cache,
            v_cache,
            p_mask,
            p_ar_mask,
            h_buf,
            id_buf,
            t0,
        )
        (
            curr_h,
            curr_id,
            finished,
            k_cache,
            v_cache,
            p_mask,
            p_ar_mask,
            h_buf,
            id_buf,
            t,
        ) = jax.lax.while_loop(cond, step, carry)

        return p_mask, p_ar_mask, h_buf, id_buf, t, k_cache, v_cache

    def sample_reasoning(self, observation: CoTObservation):
        _, _, _, logits, t, _, _ = self._sample_reasoning_tokens(observation)
        return logits, t

        # output_tokens = self._sample_reasoning_tokens(observation)
        # return output_tokens
