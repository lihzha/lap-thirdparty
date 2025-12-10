from functools import cache
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import openpi.models.model as _model
from openpi.models.model import Observation
import openpi.models.pi0 as _pi0
import openpi.models.pi0_fast as _pi0_fast
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
from typing_extensions import override

import openpi_cot.models.backbones.gemma_moe as _gemma
from openpi_cot.models.label_smoothing import create_digit_smoothing_kernel
from openpi_cot.models.label_smoothing import get_digit_to_token_mapping
from openpi_cot.models.model_adapter import CoTObservation
from openpi_cot.models.model_adapter import preprocess_observation
import openpi_cot.models.pi_cot_config as _pi_cot_config

logger = logging.getLogger("openpi")
PALIGEMMA_VOCAB_SIZE = 257_152


@cache
def _get_cached_smoothing_kernel(sigma: float, support: int) -> jnp.ndarray:
    """Create smoothing kernel once per (sigma, support) pair and reuse."""
    digit_to_token = get_digit_to_token_mapping()
    return create_digit_smoothing_kernel(
        vocab_size=PALIGEMMA_VOCAB_SIZE,
        digit_to_token_id=digit_to_token,
        sigma=sigma,
        support=support,
    )


def cross_entropy_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    mask: jnp.ndarray | None = None,
    axis: int = -1,
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

    Returns
    -------
      If per_example=False (default): scalar mean.
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
    return total / denom


class PiCoT(_pi0.Pi0):
    EOS_TOKEN: int = 1

    def __init__(self, config: _pi_cot_config.PiCoTConfig, rngs: nnx.Rngs):
        _model.BaseModel.__init__(self, config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.verbose_mode = config.verbose_mode
        self.aug_wrist_image = config.aug_wrist_image
        self.image_keys = config.image_keys
        # Loss/control knobs
        self.enable_action_training = bool(config.enable_action_training)
        self.enable_langact_training = bool(config.enable_langact_training)
        self.enable_prediction_training = bool(config.enable_prediction_training)
        self.enable_vqa_training = bool(config.enable_vqa_training)
        self.language_loss_weight = float(getattr(config, "language_loss_weight", 1.0))
        self.action_loss_weight = float(getattr(config, "action_loss_weight", 1.0))
        self.prediction_loss_weight = float(getattr(config, "prediction_loss_weight", 0.2))
        self.vqa_loss_weight = float(getattr(config, "vqa_loss_weight", 0.1))

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
                stop_action_to_vlm_grad=config.stop_action_to_vlm_grad,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        # Conditionally set the positional embedding type for the image model

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

        # img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
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

        # Label smoothing for number tokens
        self.enable_number_label_smoothing = getattr(config, "enable_number_label_smoothing", False)
        if self.enable_number_label_smoothing:
            self.label_smoothing_sigma = getattr(config, "label_smoothing_sigma", 1.0)
            self.label_smoothing_support = getattr(config, "label_smoothing_support", 3)
            logger.info(
                f"Label smoothing enabled for units digits: sigma={self.label_smoothing_sigma}, support={self.label_smoothing_support}"
            )
            # Precompute once on the host to avoid caching a traced value when first accessed inside JIT.
            _ = _get_cached_smoothing_kernel(self.label_smoothing_sigma, self.label_smoothing_support)
        else:
            self.label_smoothing_sigma = None
            self.label_smoothing_support = None

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @property
    def smoothing_kernel(self) -> jnp.ndarray | None:
        """Lazily construct and cache the smoothing kernel outside model state."""
        if not self.enable_number_label_smoothing:
            return None
        return _get_cached_smoothing_kernel(self.label_smoothing_sigma, self.label_smoothing_support)

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

            num_patches = image_tokens.shape[1]
            # Reshape: [b, t*num_patches, d]
            image_tokens = image_tokens.reshape(b, t * num_patches, -1)
            total_seq_len = t * num_patches

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=total_seq_len,
                )
            )
            # All image tokens attend to each other (no autoregressive masking)
            _img_ar_masks += [False] * total_seq_len

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        img_ar_mask = jnp.array(_img_ar_masks)
        img_ar_mask = einops.repeat(img_ar_mask, "s -> b s", b=tokens.shape[0])

        return tokens, input_mask, img_ar_mask

    @at.typecheck
    def embed_prefix(
        self,
        obs: CoTObservation | Observation,
        num_frames: int | None = None,
        precomputed_img_embeddings: tuple[
            at.Float[at.Array, "b s_in emb"], at.Bool[at.Array, "b s_in"], at.Bool[at.Array, "b s_in"]
        ]
        | None = None,
    ) -> tuple[
        at.Float[at.Array, "b s_out emb"],
        at.Bool[at.Array, "b s_out"],
        at.Bool[at.Array, "b s_out"],
    ]:
        """Embed images and text for the prefix.

        Args:
            obs: Observation containing images and tokenized text
            num_frames: If specified, only use first num_frames of images. If None, use all frames.
            precomputed_img_embeddings: Optional pre-computed image embeddings to avoid re-encoding.
                                       Tuple of (img_tokens, img_mask, img_ar_mask).

        Returns:
            (prefix_tokens, prefix_mask, prefix_ar_mask)
        """
        # Get image embeddings (use precomputed if available)
        if precomputed_img_embeddings is not None:
            img_tokens, img_mask, img_ar_mask = precomputed_img_embeddings
        else:
            img_tokens, img_mask, img_ar_mask = self._embed_images(obs, num_frames)

        if obs.tokenized_prompt is not None:
            text_tokens, text_mask, text_ar_mask = (
                self.PaliGemma.llm(obs.tokenized_prompt, method="embed"),
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

    def _compute_token_accuracy_metrics(
        self,
        predictions: at.Int[at.Array, "b s"],
        labels: at.Int[at.Array, "b s"],
        token_mask: at.Bool[at.Array, "b s"],
        critical_mask: at.Bool[at.Array, "b s"] | None = None,
        number_mask: at.Bool[at.Array, "b s"] | None = None,
        direction_mask: at.Bool[at.Array, "b s"] | None = None,
    ) -> dict[str, at.Array]:
        """Compute token accuracy metrics including critical, number, and direction tokens.

        Args:
            predictions: Predicted token IDs [b, s]
            labels: Ground truth token IDs [b, s]
            token_mask: Mask indicating which tokens to include [b, s]
            critical_mask: Optional mask for critical tokens [b, s]
            number_mask: Optional mask for number tokens [b, s]
            direction_mask: Optional mask for direction tokens [b, s]

        Returns:
            Dictionary containing accuracy metrics
        """
        metrics = {}

        # Overall token accuracy
        correct = (predictions == labels).astype(jnp.float32)
        masked_correct = correct * token_mask
        num_tokens = jnp.maximum(token_mask.sum(), 1.0)
        metrics["token_accuracy"] = masked_correct.sum() / num_tokens

        # Critical token accuracy
        if critical_mask is not None:
            critical_correct = correct * critical_mask
            # Scalar (micro-averaged)
            num_critical = jnp.maximum(critical_mask.sum(), 1.0)
            metrics["critical_token_accuracy"] = critical_correct.sum() / num_critical
            # Per-sample
            per_sample_critical_correct = critical_correct.sum(axis=-1)
            per_sample_num_critical = critical_mask.sum(axis=-1)
            metrics["per_sample_critical_correct"] = per_sample_critical_correct
            metrics["per_sample_critical_total"] = per_sample_num_critical

        # Number token accuracy
        if number_mask is not None:
            number_correct = correct * number_mask
            # Scalar (micro-averaged)
            num_number = jnp.maximum(number_mask.sum(), 1.0)
            metrics["number_token_accuracy"] = number_correct.sum() / num_number
            per_sample_number_correct = number_correct.sum(axis=-1)
            per_sample_num_number = number_mask.sum(axis=-1)
            metrics["per_sample_number_correct"] = per_sample_number_correct
            metrics["per_sample_number_total"] = per_sample_num_number

        # Direction token accuracy
        if direction_mask is not None:
            direction_correct = correct * direction_mask
            # Scalar (micro-averaged)
            num_direction = jnp.maximum(direction_mask.sum(), 1.0)
            metrics["direction_token_accuracy"] = direction_correct.sum() / num_direction
            per_sample_direction_correct = direction_correct.sum(axis=-1)
            per_sample_num_direction = direction_mask.sum(axis=-1)
            metrics["per_sample_direction_correct"] = per_sample_direction_correct
            metrics["per_sample_direction_total"] = per_sample_num_direction

        return metrics

    def _compute_cross_entropy_with_metrics(
        self,
        logits: at.Float[at.Array, "b s v"],
        labels: at.Int[at.Array, "b s"],
        token_mask: at.Bool[at.Array, "b s"],
        critical_mask: at.Bool[at.Array, "b s"] | None = None,
        number_mask: at.Bool[at.Array, "b s"] | None = None,
        direction_mask: at.Bool[at.Array, "b s"] | None = None,
        units_number_mask: at.Bool[at.Array, "b s"] | None = None,
        digit_values: at.Int[at.Array, "b s"] | None = None,
        smoothing_kernel: at.Float[at.Array, "10 v"] | None = None,
        verbose_mode: bool = False,
        return_predictions: bool = False,
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        """Compute cross-entropy loss and associated accuracy metrics.

        Args:
            logits: Model predictions [b, s, vocab_size]
            labels: Ground truth token IDs [b, s]
            token_mask: Mask indicating which tokens to include [b, s]
            critical_mask: Optional mask for critical tokens [b, s] (only used if verbose_mode=True)
            number_mask: Optional mask for number tokens [b, s] (only used if verbose_mode=True)
            direction_mask: Optional mask for direction tokens [b, s] (only used if verbose_mode=True)
            units_number_mask: Optional mask for units digit tokens [b, s]
            digit_values: Optional digit values (0-9) for number tokens [b, s]
            smoothing_kernel: Optional smoothing kernel [10, vocab_size] for label smoothing
            verbose_mode: Whether to compute detailed accuracy metrics (requires critical/number/direction masks)
            return_predictions: Whether to return predictions and labels in metrics (independent of verbose_mode)

        Returns:
            (loss, metrics_dict)
        """
        metrics = {}

        # Check if label smoothing should be applied
        # Note: We check only for None to keep the condition static (not traced)
        # If units_number_mask is all False, the masking logic below handles it correctly
        use_label_smoothing = (
            units_number_mask is not None and digit_values is not None and smoothing_kernel is not None
        )

        if use_label_smoothing:
            # Memory-efficient label smoothing implementation
            # Instead of creating full [b, s, V] arrays, we compute losses separately
            # and blend them, only materializing smoothed distributions for units digits

            log_probs = nnx.log_softmax(logits, axis=-1)  # [b, s, V]

            # Identify which tokens need smoothing (units digits with valid digit values)
            valid_digit_mask = digit_values >= 0  # [b, s]
            smoothed_token_mask = jnp.logical_and(units_number_mask, valid_digit_mask)  # [b, s]

            # 1. Compute hard-target loss for non-smoothed tokens (memory efficient)
            gather_idx = jnp.expand_dims(labels.astype(jnp.int32), axis=-1)  # [b, s, 1]
            gold_logp = jnp.take_along_axis(log_probs, gather_idx, axis=-1).squeeze(-1)  # [b, s]
            hard_loss = -gold_logp  # [b, s]

            # 2. Compute smoothed loss for units digits (only for those specific tokens)
            # Get smoothed distributions for units digits
            clipped_digit_values = jnp.clip(digit_values, 0, 9)
            smoothed_dists = smoothing_kernel[clipped_digit_values]  # [b, s, V]
            # KL divergence: -sum(p_target * log(p_model))
            smoothed_loss = -jnp.sum(smoothed_dists * log_probs, axis=-1)  # [b, s]

            # 3. Blend: use smoothed loss for units digits, hard loss for everything else
            # This avoids creating full vocab-sized target distributions
            per_token_loss = jnp.where(smoothed_token_mask, smoothed_loss, hard_loss)  # [b, s]

            # Apply token mask and reduce
            reduce_axes = tuple(range(1, per_token_loss.ndim))
            masked_loss = per_token_loss * token_mask
            denom = jnp.maximum(token_mask.sum(axis=reduce_axes), 1)
            per_sample_loss = masked_loss.sum(axis=reduce_axes) / denom
        else:
            logp = jax.nn.log_softmax(logits, axis=-1)
            token_pplx = jnp.sum(labels * logp, axis=-1)
            # Standard hard target loss
            per_sample_loss = -jnp.sum(token_pplx * token_mask, axis=-1) / jnp.clip(jnp.sum(token_mask, -1), 1)

        # Return predictions if requested (independently of verbose_mode)
        # IMPORTANT: Predictions are ONLY returned when return_predictions=True
        if return_predictions:
            predictions = jnp.argmax(logits, axis=-1)
            metrics["predictions"] = predictions
            metrics["labels"] = labels
            metrics["token_mask"] = token_mask

        # Compute detailed accuracy metrics if verbose_mode is enabled
        # NOTE: When verbose_mode=True but return_predictions=False, predictions are
        # computed internally for accuracy metrics but NOT added to the output metrics
        if verbose_mode:
            # Reuse predictions if already computed (when return_predictions=True),
            # otherwise compute them temporarily for accuracy calculation only
            predictions = metrics.get("predictions", jnp.argmax(logits, axis=-1))
            accuracy_metrics = self._compute_token_accuracy_metrics(
                predictions=predictions,
                labels=labels,
                token_mask=token_mask,
                critical_mask=critical_mask,
                number_mask=number_mask,
                direction_mask=direction_mask,
            )
            # Only add accuracy metrics, not predictions (unless already added above)
            metrics.update(accuracy_metrics)

        return per_sample_loss, metrics

    def _compute_language_loss(
        self,
        observation: CoTObservation | Observation,
        prefix_pre_logits: at.Float[at.Array, "b s emb"],
        sample_mask: at.Bool[at.Array, "b"] | None = None,
        *,
        verbose_mode: bool = False,
        return_predictions: bool = False,
        loss_name: str = "lang_loss",
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        """Compute language/reasoning loss given precomputed prefix pre-logits."""

        effective_sample_mask = sample_mask if sample_mask is not None else observation.sample_mask

        targets = jax.nn.one_hot(
            observation.tokenized_prompt[:, 1:],
            PALIGEMMA_VOCAB_SIZE,
        )

        # Drop final prefix token (no next-token target) then align to text targets.
        pre_logits = prefix_pre_logits[:, :-1]
        pre_logits = pre_logits[:, -targets.shape[1] :]
        logits, _, _ = self.PaliGemma.llm(pre_logits=pre_logits)

        loss_mask = jnp.logical_and(observation.tokenized_langact_mask[:, 1:], observation.tokenized_prompt_mask[:, 1:])
        if effective_sample_mask is not None:
            ex_mask = jnp.asarray(effective_sample_mask)[..., None]
            loss_mask = loss_mask * ex_mask
        else:
            ex_mask = None

        def prepare_mask(mask):
            if mask is None:
                return None
            shifted_mask = mask[:, 1:]
            if ex_mask is not None:
                return shifted_mask * ex_mask
            return shifted_mask

        units_mask_shifted = prepare_mask(getattr(observation, "units_number_token_mask", None))
        digit_values = getattr(observation, "digit_values", None)
        digit_values_shifted = digit_values[:, 1:] if digit_values is not None else None

        if verbose_mode:
            critical_mask = prepare_mask(observation.critical_token_mask)
            number_mask = prepare_mask(observation.number_token_mask)
            direction_mask = prepare_mask(observation.direction_token_mask)
        else:
            critical_mask = number_mask = direction_mask = None

        smoothing_kernel = None
        if self.enable_number_label_smoothing:
            smoothing_kernel = self.smoothing_kernel.astype(logits.dtype)

        per_sample_loss, raw_metrics = self._compute_cross_entropy_with_metrics(
            logits=logits,
            labels=targets,
            token_mask=loss_mask,
            critical_mask=critical_mask,
            number_mask=number_mask,
            direction_mask=direction_mask,
            units_number_mask=units_mask_shifted,
            digit_values=digit_values_shifted,
            smoothing_kernel=smoothing_kernel,
            verbose_mode=verbose_mode,
            return_predictions=return_predictions,
        )

        metrics = {loss_name: jnp.mean(per_sample_loss)}

        if return_predictions or verbose_mode:
            metric_rename_map = {
                "token_accuracy": "token_accuracy",
                "critical_token_accuracy": "critical_token_accuracy",
                "number_token_accuracy": "number_token_accuracy",
                "direction_token_accuracy": "direction_token_accuracy",
            }
            for key, value in raw_metrics.items():
                if key in ["predictions", "labels", "token_mask"]:
                    metrics[key] = value
                else:
                    metrics[metric_rename_map.get(key, key)] = value

        return per_sample_loss, metrics

    def _compute_action_loss(
        self,
        suffix_out: at.Float[at.Array, "b s emb"],
        u_t: at.Float[at.Array, "b ah ad"],
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        """Compute action diffusion loss from suffix activations."""

        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        per_sample_action_loss = jnp.mean(jnp.square(v_t - u_t), axis=(-1, -2))
        metrics = {"action_loss": jnp.mean(per_sample_action_loss)}
        return per_sample_action_loss, metrics

    def _build_prefix_action_mask(
        self,
        prefix_mask: at.Bool[at.Array, "b s"],
        observation: CoTObservation | Observation,
    ) -> at.Bool[at.Array, "b s"]:
        if observation.tokenized_langact_mask is None:
            return prefix_mask
        img_seq_len = prefix_mask.shape[1] - observation.tokenized_langact_mask.shape[1]
        langact_mask_full = jnp.concatenate(
            [
                jnp.zeros((observation.tokenized_langact_mask.shape[0], img_seq_len), dtype=bool),
                observation.tokenized_langact_mask,
            ],
            axis=1,
        )
        return jnp.logical_and(prefix_mask, jnp.logical_not(langact_mask_full))

    def _prepare_action_suffix(
        self,
        observation: CoTObservation | Observation,
        actions: _model.Actions,
        noise_rng: at.KeyArrayLike,
        time_rng: at.KeyArrayLike,
    ) -> dict[str, at.Array]:
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        suffix_ar_mask = einops.repeat(suffix_ar_mask, "s -> b s", b=suffix_tokens.shape[0])
        return {
            "suffix_tokens": suffix_tokens,
            "suffix_mask": suffix_mask,
            "suffix_ar_mask": suffix_ar_mask,
            "adarms_cond": adarms_cond,
            "u_t": u_t,
        }

    def _build_combined_attention_mask(
        self,
        prefix_mask: at.Bool[at.Array, "b p"],
        prefix_ar_mask: at.Bool[at.Array, "b p"],
        prefix_mask_action: at.Bool[at.Array, "b p"],
        suffix_mask: at.Bool[at.Array, "b s"] | None,
        suffix_ar_mask: at.Bool[at.Array, "b s"] | None,
    ) -> at.Bool[at.Array, "b _t _s"]:
        prefix_attn = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        if suffix_mask is None or suffix_ar_mask is None:
            return prefix_attn

        batch_size, prefix_len = prefix_mask.shape
        suffix_len = suffix_mask.shape[1]
        combined = jnp.zeros((batch_size, prefix_len + suffix_len, prefix_len + suffix_len), dtype=bool)
        combined = combined.at[:, :prefix_len, :prefix_len].set(prefix_attn)

        prefix_ar_mask_action = jnp.zeros_like(prefix_mask_action, dtype=bool)
        input_mask = jnp.concatenate([prefix_mask_action, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask_action, suffix_ar_mask], axis=1)
        action_mask = _pi0.make_attn_mask(input_mask, ar_mask)
        combined = combined.at[:, prefix_len:, :].set(action_mask[:, prefix_len:, :])
        return combined

    def _build_combined_positions(
        self,
        prefix_mask: at.Bool[at.Array, "b p"],
        prefix_mask_action: at.Bool[at.Array, "b p"],
        suffix_mask: at.Bool[at.Array, "b s"] | None,
    ) -> at.Int[at.Array, "b _t"]:
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        if suffix_mask is None:
            return prefix_positions.astype(jnp.int32)
        suffix_positions = jnp.sum(prefix_mask_action, axis=-1, keepdims=True) + jnp.cumsum(suffix_mask, axis=-1) - 1
        combined = jnp.concatenate([prefix_positions, suffix_positions], axis=1)
        return combined.astype(jnp.int32)

    def _compute_sample_specific_metrics(
        self,
        per_sample_loss: at.Float[at.Array, "b"],
        lang_metrics: dict[str, at.Array],
        sample_mask: at.Bool[at.Array, "b"],
        prefix: str,
        *,
        verbose_mode: bool = False,
    ) -> dict[str, at.Array]:
        """Compute comprehensive metrics for a specific subset of samples (pred or langact).

        Args:
            per_sample_loss: Per-sample losses [b]
            lang_metrics: Metrics dict from _compute_language_loss
            sample_mask: Boolean mask indicating which samples to include [b]
            prefix: Prefix for metric names (e.g., "pred_" or "langact_")
            verbose_mode: Whether to compute detailed metrics

        Returns:
            Dictionary containing all metrics for this sample subset
        """
        metrics = {}

        # Masked per-sample loss
        masked_loss = per_sample_loss * sample_mask
        num_samples = jnp.maximum(jnp.sum(sample_mask), 1.0)

        # Average loss for this subset
        metrics[f"{prefix}loss"] = jnp.sum(masked_loss) / num_samples

        if verbose_mode:
            # Per-sample losses (for dataset-level micro-averaging)
            metrics[f"{prefix}per_sample_loss"] = masked_loss

            # Critical token metrics
            if "per_sample_critical_correct" in lang_metrics:
                critical_correct = lang_metrics["per_sample_critical_correct"] * sample_mask
                critical_total = lang_metrics["per_sample_critical_total"] * sample_mask
                num_critical_tokens = jnp.maximum(jnp.sum(critical_total), 1.0)

                # Average critical token accuracy
                metrics[f"{prefix}critical_token_accuracy"] = jnp.sum(critical_correct) / num_critical_tokens

                # Per-sample counts (for dataset-level micro-averaging)
                metrics[f"{prefix}per_sample_critical_correct"] = critical_correct
                metrics[f"{prefix}per_sample_critical_total"] = critical_total

            # Number token metrics
            if "per_sample_number_correct" in lang_metrics:
                number_correct = lang_metrics["per_sample_number_correct"] * sample_mask
                number_total = lang_metrics["per_sample_number_total"] * sample_mask
                num_number_tokens = jnp.maximum(jnp.sum(number_total), 1.0)

                # Average number token accuracy
                metrics[f"{prefix}number_token_accuracy"] = jnp.sum(number_correct) / num_number_tokens

                # Per-sample counts (for dataset-level micro-averaging)
                metrics[f"{prefix}per_sample_number_correct"] = number_correct
                metrics[f"{prefix}per_sample_number_total"] = number_total

            # Direction token metrics
            if "per_sample_direction_correct" in lang_metrics:
                direction_correct = lang_metrics["per_sample_direction_correct"] * sample_mask
                direction_total = lang_metrics["per_sample_direction_total"] * sample_mask
                num_direction_tokens = jnp.maximum(jnp.sum(direction_total), 1.0)

                # Average direction token accuracy
                metrics[f"{prefix}direction_token_accuracy"] = jnp.sum(direction_correct) / num_direction_tokens

                # Per-sample counts (for dataset-level micro-averaging)
                metrics[f"{prefix}per_sample_direction_correct"] = direction_correct
                metrics[f"{prefix}per_sample_direction_total"] = direction_total

        return metrics

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: CoTObservation | Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        stage_config: dict | None = None,
        verbose_mode: bool | None = None,
    ) -> dict[str, at.Array]:
        preprocess_rng, _, noise_rng, time_rng = jax.random.split(rng, 4)

        # Use passed verbose_mode if provided, otherwise use class attribute
        effective_verbose_mode = verbose_mode if verbose_mode is not None else self.verbose_mode

        # Determine batch size
        batch_size = observation.tokenized_prompt.shape[0]

        # Compute VQA mask first (before preprocessing) to skip augmentation for VQA samples
        vqa_mask = None
        if self.enable_vqa_training and hasattr(observation, "is_vqa_sample") and observation.is_vqa_sample is not None:
            vqa_mask = jnp.asarray(observation.is_vqa_sample, dtype=bool)

        # Preprocess observation (will skip augmentation for VQA samples if vqa_mask is provided)
        observation = preprocess_observation(
            preprocess_rng,
            observation,
            train=train,
            image_keys=self.image_keys,
            aug_wrist_image=self.aug_wrist_image,
            vqa_mask=vqa_mask,
        )

        # Encode images (only first frame needed since prediction is handled at dataset level)
        img_tokens_first, img_mask_first, img_ar_mask_first = self._embed_images(observation, num_frames=1)

        # Build prefix for langact/action losses (first frame + text)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(
            observation, num_frames=1, precomputed_img_embeddings=(img_tokens_first, img_mask_first, img_ar_mask_first)
        )

        suffix_inputs = (
            self._prepare_action_suffix(observation, actions, noise_rng, time_rng)
            if self.enable_action_training
            else None
        )
        prefix_mask_action = (
            self._build_prefix_action_mask(prefix_mask, observation) if suffix_inputs is not None else prefix_mask
        )
        combined_mask = self._build_combined_attention_mask(
            prefix_mask,
            prefix_ar_mask,
            prefix_mask_action,
            suffix_inputs["suffix_mask"] if suffix_inputs is not None else None,
            suffix_inputs["suffix_ar_mask"] if suffix_inputs is not None else None,
        )
        combined_positions = self._build_combined_positions(
            prefix_mask, suffix_inputs["suffix_mask"] if suffix_inputs is not None else None, prefix_mask_action
        )
        embedded_inputs = [prefix_tokens, suffix_inputs["suffix_tokens"] if suffix_inputs is not None else None]
        adarms_cond = [None, suffix_inputs["adarms_cond"] if suffix_inputs is not None else None]
        prefix_pre_logits, _, gemma_out = self.PaliGemma.llm(
            embedded_inputs,
            mask=combined_mask,
            positions=combined_positions,
            adarms_cond=adarms_cond,
            return_prelogits=True,
            deterministic=self.deterministic,
        )
        suffix_out = gemma_out["encoded"][1] if suffix_inputs is not None else None

        metrics = {}
        total_per_sample_loss = 0

        if self.enable_langact_training:
            combined_langact_mask = observation.sample_mask
            lang_loss, lang_metrics = self._compute_language_loss(
                observation,
                prefix_pre_logits,
                sample_mask=combined_langact_mask,
                verbose_mode=effective_verbose_mode,
            )

            if self.enable_vqa_training or self.enable_prediction_training:
                if vqa_mask is None:
                    vqa_mask = jnp.zeros(batch_size, dtype=bool)
                pred_mask = (
                    jnp.asarray(observation.is_prediction_sample, dtype=bool)
                    if self.enable_prediction_training
                    else jnp.zeros(batch_size, dtype=bool)
                )
                lang_mask = jnp.logical_not(jnp.logical_or(vqa_mask, pred_mask))

                vqa_mask = jnp.logical_and(vqa_mask, combined_langact_mask)
                pred_mask = jnp.logical_and(pred_mask, combined_langact_mask)
                lang_mask = jnp.logical_and(lang_mask, combined_langact_mask)

                if self.enable_vqa_training:
                    metrics.update(
                        self._compute_sample_specific_metrics(
                            per_sample_loss=lang_loss,
                            lang_metrics=lang_metrics,
                            sample_mask=vqa_mask,
                            prefix="vqa_",
                            verbose_mode=effective_verbose_mode,
                        )
                    )
                if self.enable_prediction_training:
                    metrics.update(
                        self._compute_sample_specific_metrics(
                            per_sample_loss=lang_loss,
                            lang_metrics=lang_metrics,
                            sample_mask=pred_mask,
                            prefix="pred_",
                            verbose_mode=effective_verbose_mode,
                        )
                    )
                metrics.update(
                    self._compute_sample_specific_metrics(
                        per_sample_loss=lang_loss,
                        lang_metrics=lang_metrics,
                        sample_mask=lang_mask,
                        prefix="langact_",
                        verbose_mode=effective_verbose_mode,
                    )
                )

                total_per_sample_loss += (
                    self.vqa_loss_weight * lang_loss * vqa_mask
                    + self.prediction_loss_weight * lang_loss * pred_mask
                    + self.language_loss_weight * lang_loss * lang_mask
                )
            else:
                metrics.update(
                    self._compute_sample_specific_metrics(
                        per_sample_loss=lang_loss,
                        lang_metrics=lang_metrics,
                        sample_mask=combined_langact_mask,
                        prefix="langact_",
                        verbose_mode=effective_verbose_mode,
                    )
                )
                total_per_sample_loss += self.language_loss_weight * lang_loss

        if suffix_out is not None and suffix_inputs is not None:
            action_loss, action_metrics = self._compute_action_loss(suffix_out, suffix_inputs["u_t"])
            total_per_sample_loss += self.action_loss_weight * action_loss
            metrics.update(action_metrics)

        # Add main metrics to dict
        if effective_verbose_mode:
            metrics["per_sample_loss"] = total_per_sample_loss

        # Compute final loss with correct normalization
        # When samples are masked out, their loss is 0 and shouldn't be counted in denominator
        if self.enable_action_training:
            # Action training applies to all samples, so use batch size normalization
            final_loss = jnp.mean(total_per_sample_loss)
        elif self.enable_langact_training and observation.sample_mask is not None:
            # Only langact training with sample masking: divide by number of active samples
            num_active_samples = jnp.maximum(jnp.sum(observation.sample_mask), 1.0)
            final_loss = jnp.sum(total_per_sample_loss) / num_active_samples
        else:
            # No masking or fallback: use mean over all samples
            final_loss = jnp.mean(total_per_sample_loss)

        return final_loss, metrics

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
        _, kv_cache, _ = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions,
            adarms_cond=[None, None],
            deterministic=self.deterministic,
        )

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

            _, _, gemma_out = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
                deterministic=self.deterministic,
            )
            suffix_out = gemma_out["encoded"][1]
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    @override
    def sample_tokens(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        max_decoding_steps: int | at.Int[at.Array, ""] = 256,
        temperature: float = 0.0,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(
            None, observation, train=False, image_keys=list(observation.images.keys())
        )

        # embed inputs
        prefix_token_embeddings, prefix_mask, prefix_ar_mask = self.embed_inputs(observation)
        prefix_attn_mask = _pi0_fast.make_attn_mask(prefix_mask, prefix_ar_mask)

        # left to right align all input token sequences
        prefix_token_embeddings, prefix_mask, prefix_attn_mask = _pi0_fast.left_to_right_align(
            prefix_token_embeddings, prefix_mask, prefix_attn_mask
        )
        prefill_size = prefix_token_embeddings.shape[1]
        prefill_len = jnp.sum(prefix_mask, axis=-1)
        prefix_start = prefill_size - prefill_len

        # first fill KV cache with a forward pass of the prefix
        # pad attention mask to set the size of the KV cache (prefill_size + max_decoding_steps)
        prefix_attn_mask = jnp.pad(prefix_attn_mask, ((0, 0), (0, 0), (0, max_decoding_steps)))
        prefix_positions = jnp.cumsum(prefix_mask, axis=-1) - 1
        prefix_logits, kv_cache, _ = self.PaliGemma.llm(
            [prefix_token_embeddings, None],
            mask=prefix_attn_mask,
            positions=prefix_positions,
            adarms_cond=[None, None],
            deterministic=self.deterministic,
        )

        # prepare decoding -- final logit decodes the first token
        last_logit = prefix_logits[:, -1:]
        output_tokens = jnp.zeros((last_logit.shape[0], max_decoding_steps))

        def step(carry):
            rng, last_logit, output_tokens, cache, _, step = carry

            # Sample token from last logit
            # Split RNG for this step
            rng, rng_step = jax.random.split(rng)
            token = jax.lax.cond(
                temperature > 0.0,
                lambda _: jax.random.categorical(rng_step, last_logit / temperature, axis=-1),
                lambda _: jnp.argmax(last_logit, axis=-1),
                operand=None,
            )
            output_tokens = _pi0_fast.put_along_last_axis(
                output_tokens, jnp.broadcast_to(step, (token.shape[0], 1)), token
            )

            # Check for early stopping --> stop if all batch elements have EOS token
            has_eos = jnp.any(token == self.EOS_TOKEN, axis=-1)
            all_eos = jnp.all(has_eos)

            # Decode one step
            token_embedding = self.PaliGemma.llm(token, method="embed")
            positions = prefill_len[:, None] + step + 1
            mask = jnp.logical_and(
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :] >= prefix_start[:, None, None],
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :]
                < (jnp.broadcast_to(prefill_size + step + 1, (prefix_start.shape[0], 1, 1))),
            )
            last_logit, kv_cache, _ = self.PaliGemma.llm(
                [token_embedding, None],
                mask=mask,
                positions=positions,
                kv_cache=cache,
                adarms_cond=[None, None],
                deterministic=self.deterministic,
            )

            return rng, last_logit, output_tokens, kv_cache, all_eos, step + 1

        def cond(carry):
            _, _, _, _, all_eos, step = carry
            return (~all_eos) & (step < max_decoding_steps)

        # Use lax.while_loop so we can jit the full decoding loop.
        _, _, output_tokens, _, _, _ = jax.lax.while_loop(
            cond, step, (rng, last_logit, output_tokens, kv_cache, False, 0)
        )
        return output_tokens
