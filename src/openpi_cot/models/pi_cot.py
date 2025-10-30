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
import openpi_cot.models.siglip as _siglip_gemma3

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
    BEGIN_IMAGE_TOKEN = 255999  # only for Gemma3
    END_IMAGE_TOKEN = 262144  # only for Gemma3
    NEW_LINE_TOKEN = 108  # only for Gemma3

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
        self.use_gemma3 = False
        self.EOS_TOKEN = 1
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
            self.use_gemma3 = True
            self.EOS_TOKEN = 106
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
                _siglip_gemma3.Module(
                    num_classes=paligemma_config.width,
                    variant="So400m/14",
                    pool_type="none",
                    scan=True,
                    dtype_mm=config.dtype,
                    posemb="learn",  # needed for size-mismatch
                    posemb_shape=(16, 16),
                )
            )
            fake_obs = config.fake_obs()
            fake_obs_image = next(iter(fake_obs.images.values()))
            img.lazy_init(fake_obs_image[:, None], train=False, rngs=rngs)
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

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    def _extract_first_frame_embeddings(
        self,
        obs: CoTObservation | Observation,
        full_image_embeddings: at.Float[at.Array, "b s emb"],
        full_image_mask: at.Bool[at.Array, "b s"],
        full_image_ar_mask: at.Bool[at.Array, "b s"],
    ) -> tuple[
        at.Float[at.Array, "b s_first emb"],
        at.Bool[at.Array, "b s_first"],
        at.Bool[at.Array, "b s_first"],
    ]:
        """Extract first-frame embeddings from full image embeddings.

        Args:
            obs: Observation containing images with shape [b, t, h, w, c]
            full_image_embeddings: Embeddings from all frames [b, total_patches, emb]
            full_image_mask: Mask for all frames
            full_image_ar_mask: AR mask for all frames

        Returns:
            (first_frame_embeddings, first_frame_mask, first_frame_ar_mask)
        """
        # Calculate patches per frame (assume all images have same patch count)
        # For 224x224 images with 14x14 patches, this is 256
        first_image = next(iter(obs.images.values()))
        _, t, h, w, c = first_image.shape
        total_frames = sum(img.shape[1] for img in obs.images.values())
        num_patches = full_image_embeddings.shape[1] // total_frames

        # Extract first frame for each image key
        first_frame_tokens = []
        first_frame_masks = []
        first_frame_ar_masks = []

        offset = 0
        for name in obs.images:
            num_frames_in_image = obs.images[name].shape[1]

            # Extract first frame for this image (first num_patches tokens)
            first_frame_tokens.append(full_image_embeddings[:, offset : offset + num_patches])
            first_frame_masks.append(full_image_mask[:, offset : offset + num_patches])
            first_frame_ar_masks.append(full_image_ar_mask[:, offset : offset + num_patches])

            # Move offset by all frames of this image
            offset += num_frames_in_image * num_patches

        # Concatenate across image keys
        img_tokens_first = jnp.concatenate(first_frame_tokens, axis=1)
        img_mask_first = jnp.concatenate(first_frame_masks, axis=1)
        img_ar_mask_first = jnp.concatenate(first_frame_ar_masks, axis=1)

        return img_tokens_first, img_mask_first, img_ar_mask_first

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
            image_flat = image.reshape(b * t, h, w, c) if not self.use_gemma3 else image
            image_tokens, _ = self.PaliGemma.img(image_flat, train=False)
            # if self.use_gemma3:
            #     image_tokens = self.resize_to_256(image_tokens)
            # image_tokens: [b*t, num_patches, d]

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

    # def resize_to_256(self, x: at.Float[at.Array, "b input_len d"]) -> at.Float[at.Array, "b 256 d"]:
    #     """Resize image token sequence to 256 tokens using average pooling.

    #     Args:
    #         x: Image tokens with shape [b, input_len, d] where input_len is a perfect square

    #     Returns:
    #         Resized tokens with shape [b, 256, d]
    #     """
    #     output_length = 256
    #     cur_length = x.shape[1]

    #     if cur_length == output_length:
    #         return x

    #     cur_width = int(cur_length**0.5)
    #     assert cur_width**2 == cur_length, f"Input length {cur_length} must be a perfect square"

    #     output_width = int(output_length**0.5)  # 16
    #     assert output_width**2 == output_length

    #     assert cur_width % output_width == 0, (
    #         f"Cannot evenly pool {cur_width}x{cur_width} to {output_width}x{output_width}"
    #     )

    #     window = cur_width // output_width

    #     # Reshape to spatial grid
    #     x = einops.rearrange(x, "b (h w) d -> b h w d", h=cur_width, w=cur_width)

    #     # Average pool using einops
    #     x = einops.reduce(x, "b (h wh) (w ww) d -> b h w d", "mean", wh=window, ww=window)

    #     return einops.rearrange(x, "b h w d -> b (h w) d")

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

    def _replace_image_placeholders(
        self,
        token_embeddings: at.Float[at.Array, "b s emb"],
        tokenized_sequence: at.Int[at.Array, "b s"],
        image_embeddings: at.Float[at.Array, "b n_img*n_patches emb"],
    ) -> at.Float[at.Array, "b s emb"]:
        """Replace placeholder tokens (-2) with actual image embeddings.

        Args:
            token_embeddings: Embeddings from tokenized sequence (includes placeholder embeddings)
            tokenized_sequence: Token IDs (includes -2 for placeholders)
            image_embeddings: Actual image embeddings from SigLIP [b, n_img*n_patches, emb]

        Returns:
            Updated embeddings with placeholders replaced
        """
        # Find placeholder positions: where token_id == -2
        is_placeholder = tokenized_sequence == -2  # [b, s]

        # Count total placeholders per batch element (should be same across batch)
        num_placeholders = jnp.sum(is_placeholder, axis=1)  # [b]

        # Create indices for scattering image embeddings into placeholder positions
        # For each batch element, we need to map placeholder positions to image embedding indices
        b, s, emb = token_embeddings.shape
        _, n_img_patches, _ = image_embeddings.shape

        # Build a mapping: for each position in sequence, what image index does it correspond to?
        # Cumulative sum of is_placeholder gives us the image token index for each placeholder
        placeholder_indices = jnp.cumsum(is_placeholder, axis=1) - 1  # [b, s], -1 to make 0-indexed

        # Clamp to valid range [0, n_img_patches)
        placeholder_indices = jnp.clip(placeholder_indices, 0, n_img_patches - 1)

        # Gather image embeddings according to placeholder indices
        # For each position, if it's a placeholder, get the corresponding image embedding
        # Otherwise, keep the original token embedding
        batch_indices = jnp.arange(b)[:, None]  # [b, 1]
        selected_image_embs = image_embeddings[batch_indices, placeholder_indices]  # [b, s, emb]

        # Replace: where is_placeholder, use image embedding; otherwise use token embedding
        result = jnp.where(
            is_placeholder[..., None],  # [b, s, 1] for broadcasting
            selected_image_embs,
            token_embeddings,
        )

        return result

    def _replace_placeholder_masks(
        self,
        text_mask: at.Bool[at.Array, "b s"],
        text_ar_mask: at.Bool[at.Array, "b s"],
        tokenized_sequence: at.Int[at.Array, "b s"],
        img_mask: at.Bool[at.Array, "b n_img"],
        img_ar_mask: at.Bool[at.Array, "b n_img"],
    ) -> tuple[at.Bool[at.Array, "b s"], at.Bool[at.Array, "b s"]]:
        """Replace placeholder positions in text masks with actual image masks.

        Args:
            text_mask: Mask from tokenized sequence (includes placeholder positions)
            text_ar_mask: AR mask from tokenized sequence (includes placeholder positions)
            tokenized_sequence: Token IDs (includes -2 for placeholders)
            img_mask: Actual image mask indicating which image tokens are valid
            img_ar_mask: Actual image AR mask for autoregressive masking

        Returns:
            (updated_mask, updated_ar_mask) with placeholder positions replaced
        """
        # Find placeholder positions: where token_id == -2
        is_placeholder = tokenized_sequence == -2  # [b, s]

        b, s = text_mask.shape
        _, n_img_patches = img_mask.shape

        # Build a mapping: for each position in sequence, what image index does it correspond to?
        placeholder_indices = jnp.cumsum(is_placeholder, axis=1) - 1  # [b, s], -1 to make 0-indexed
        placeholder_indices = jnp.clip(placeholder_indices, 0, n_img_patches - 1)

        # Gather image masks according to placeholder indices
        batch_indices = jnp.arange(b)[:, None]  # [b, 1]
        selected_img_mask = img_mask[batch_indices, placeholder_indices]  # [b, s]
        selected_img_ar_mask = img_ar_mask[batch_indices, placeholder_indices]  # [b, s]

        # Replace: where is_placeholder, use image mask; otherwise use text mask
        updated_mask = jnp.where(is_placeholder, selected_img_mask, text_mask)
        updated_ar_mask = jnp.where(is_placeholder, selected_img_ar_mask, text_ar_mask)

        return updated_mask, updated_ar_mask

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

        # Build prefix using appropriate strategy for model variant
        if self.use_gemma3 and obs.tokenized_prompt is not None:
            return self._build_prefix_gemma3(obs, img_tokens, img_mask, img_ar_mask)
        return self._build_prefix_legacy(obs, img_tokens, img_mask, img_ar_mask)

    def _prepare_token_mask(
        self,
        langact_mask: at.Bool[at.Array, "b s"],
        pad_mask: at.Bool[at.Array, "b s"],
        sample_mask: at.Bool[at.Array, "b"] | None = None,
    ) -> at.Bool[at.Array, "b s"]:
        """Prepare token mask by combining langact, padding, and sample masks.

        Args:
            langact_mask: Mask indicating which tokens are part of language/action
            pad_mask: Mask indicating which tokens are not padding
            sample_mask: Optional per-sample mask for batch-level filtering

        Returns:
            Combined token mask
        """
        token_mask = jnp.logical_and(langact_mask, pad_mask)
        if sample_mask is not None:
            ex_mask = jnp.asarray(sample_mask)[..., None]
            token_mask = token_mask * ex_mask
        return token_mask

    def _compute_token_accuracy_metrics(
        self,
        predictions: at.Int[at.Array, "b s"],
        labels: at.Int[at.Array, "b s"],
        token_mask: at.Bool[at.Array, "b s"],
        critical_mask: at.Bool[at.Array, "b s"] | None = None,
        number_mask: at.Bool[at.Array, "b s"] | None = None,
        direction_mask: at.Bool[at.Array, "b s"] | None = None,
        train: bool = False,
    ) -> dict[str, at.Array]:
        """Compute token accuracy metrics including critical, number, and direction tokens.

        Args:
            predictions: Predicted token IDs [b, s]
            labels: Ground truth token IDs [b, s]
            token_mask: Mask indicating which tokens to include [b, s]
            critical_mask: Optional mask for critical tokens [b, s]
            number_mask: Optional mask for number tokens [b, s]
            direction_mask: Optional mask for direction tokens [b, s]
            train: If True, compute per-sample metrics for dataset tracking

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
            # Per-sample (only during training for dataset tracking)
            if train:
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
            # Per-sample (only during training for dataset tracking)
            if train:
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
            # Per-sample (only during training for dataset tracking)
            if train:
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
        train: bool = False,
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        """Compute cross-entropy loss and associated accuracy metrics.

        Args:
            logits: Model predictions [b, s, vocab_size]
            labels: Ground truth token IDs [b, s]
            token_mask: Mask indicating which tokens to include [b, s]
            critical_mask: Optional mask for critical tokens [b, s]
            number_mask: Optional mask for number tokens [b, s]
            direction_mask: Optional mask for direction tokens [b, s]
            train: If True, return per-sample loss and compute per-sample metrics

        Returns:
            (loss, metrics_dict)
        """
        metrics = {}

        # Compute cross-entropy loss
        loss = cross_entropy_loss(
            logits,
            labels,
            mask=token_mask,
            axis=-1,
            train=True,
            per_example=True,
        )

        # Compute accuracy metrics
        predictions = jnp.argmax(logits, axis=-1)
        accuracy_metrics = self._compute_token_accuracy_metrics(
            predictions=predictions,
            labels=labels,
            token_mask=token_mask,
            critical_mask=critical_mask,
            number_mask=number_mask,
            direction_mask=direction_mask,
            train=train,
        )
        metrics.update(accuracy_metrics)

        return loss, metrics

    def _forward_language_model(
        self,
        prefix_tokens: at.Float[at.Array, "b s emb"],
        prefix_mask: at.Bool[at.Array, "b s"],
        prefix_ar_mask: at.Bool[at.Array, "b s"],
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Array]:
        """Forward pass through language model with attention masking.

        Args:
            prefix_tokens: Input token embeddings
            prefix_mask: Input mask indicating valid tokens
            prefix_ar_mask: Autoregressive mask for causal attention

        Returns:
            (output_embeddings, kv_cache)
        """
        attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=attn_mask, positions=positions)
        return prefix_out, kv_cache

    def _build_prefix_gemma3(
        self,
        obs: CoTObservation | Observation,
        img_tokens: at.Float[at.Array, "b s_img emb"],
        img_mask: at.Bool[at.Array, "b s_img"],
        img_ar_mask: at.Bool[at.Array, "b s_img"],
    ) -> tuple[
        at.Float[at.Array, "b s_out emb"],
        at.Bool[at.Array, "b s_out"],
        at.Bool[at.Array, "b s_out"],
    ]:
        """Build prefix for Gemma3 models using placeholder replacement.

        Args:
            obs: Observation containing tokenized prompt with placeholders
            img_tokens: Precomputed image embeddings
            img_mask: Image mask
            img_ar_mask: Image autoregressive mask

        Returns:
            (tokens, input_mask, ar_mask)
        """
        # Embed the full tokenized sequence (including placeholders)
        text_tokens, text_mask, text_ar_mask = self._embed_text(
            obs.tokenized_prompt,
            obs.tokenized_prompt_mask,
            obs.tokenized_langact_mask,
        )

        # Replace placeholder embeddings with actual image embeddings
        tokens = self._replace_image_placeholders(text_tokens, obs.tokenized_prompt, img_tokens)

        # Replace placeholder masks with actual image masks
        if text_ar_mask is None:
            text_ar_mask = jnp.zeros_like(text_mask, dtype=bool)
        input_mask, ar_mask = self._replace_placeholder_masks(
            text_mask, text_ar_mask, obs.tokenized_prompt, img_mask, img_ar_mask
        )

        return tokens, input_mask, ar_mask

    def _build_prefix_legacy(
        self,
        obs: CoTObservation | Observation,
        img_tokens: at.Float[at.Array, "b s_img emb"],
        img_mask: at.Bool[at.Array, "b s_img"],
        img_ar_mask: at.Bool[at.Array, "b s_img"],
    ) -> tuple[
        at.Float[at.Array, "b s_out emb"],
        at.Bool[at.Array, "b s_out"],
        at.Bool[at.Array, "b s_out"],
    ]:
        """Build prefix for legacy models by concatenating image and text embeddings.

        Args:
            obs: Observation containing tokenized prompt
            img_tokens: Precomputed image embeddings
            img_mask: Image mask
            img_ar_mask: Image autoregressive mask

        Returns:
            (tokens, input_mask, ar_mask)
        """
        # Add language (aka tokenized inputs)
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

    def _compute_language_loss(
        self,
        observation: CoTObservation | Observation,
        prefix_tokens: at.Float[at.Array, "b s emb"],
        prefix_mask: at.Bool[at.Array, "b s"],
        prefix_ar_mask: at.Bool[at.Array, "b s"],
        train: bool,
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array], at.Float[at.Array, ""]]:
        """Compute language/reasoning cross-entropy loss and accuracy metrics.

        Args:
            observation: Observation containing tokenized prompts and masks
            prefix_tokens: Prefix embeddings (images + text)
            prefix_mask: Prefix attention mask
            prefix_ar_mask: Prefix autoregressive mask
            train: If True, return per-sample loss and metrics

        Returns:
            (loss, metrics, token_accuracy_scalar)
        """
        # Forward pass
        prefix_out, _ = self._forward_language_model(prefix_tokens, prefix_mask, prefix_ar_mask)

        # Predict next tokens over the reasoning span
        shift_labels = observation.tokenized_prompt[:, 1:]
        max_len = observation.tokenized_langact_mask.shape[1]
        shift_tokens = prefix_out[:, -max_len:-1, :]
        shift_logits = self.PaliGemma.llm(shift_tokens, method="decode")

        # Prepare token mask
        token_mask = self._prepare_token_mask(
            observation.tokenized_langact_mask[:, 1:],
            observation.tokenized_prompt_mask[:, 1:],
            observation.sample_mask,
        )

        # Prepare additional masks for accuracy computation
        ex_mask = jnp.asarray(observation.sample_mask)[..., None] if observation.sample_mask is not None else None
        critical_mask = (
            observation.crictical_token_mask[:, 1:] * ex_mask
            if ex_mask is not None
            else observation.crictical_token_mask[:, 1:]
        )
        number_mask = (
            observation.number_token_mask[:, 1:] * ex_mask
            if observation.number_token_mask is not None and ex_mask is not None
            else observation.number_token_mask[:, 1:]
            if observation.number_token_mask is not None
            else None
        )
        direction_mask = (
            observation.direction_token_mask[:, 1:] * ex_mask
            if observation.direction_token_mask is not None and ex_mask is not None
            else observation.direction_token_mask[:, 1:]
            if observation.direction_token_mask is not None
            else None
        )

        # Compute loss and metrics
        loss, metrics = self._compute_cross_entropy_with_metrics(
            logits=shift_logits,
            labels=shift_labels,
            token_mask=token_mask,
            critical_mask=critical_mask,
            number_mask=number_mask,
            direction_mask=direction_mask,
            train=train,
        )

        # Store loss in metrics
        if train:
            metrics["lang_loss"] = loss
        else:
            metrics["lang_loss"] = jnp.mean(loss)

        # Extract scalar token accuracy for backward compatibility
        token_accuracy = metrics.get("token_accuracy", jnp.array(0.0))

        return loss, metrics, token_accuracy

    def _compute_prediction_loss(
        self,
        observation: CoTObservation | Observation,
        img_tokens_all: at.Float[at.Array, "b s_img emb"],
        img_mask_all: at.Bool[at.Array, "b s_img"],
        img_ar_mask_all: at.Bool[at.Array, "b s_img"],
        train: bool,
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        """Compute prediction cross-entropy loss using all video frames.

        Args:
            observation: Observation containing tokenized predictions
            img_tokens_all: All-frame image embeddings
            img_mask_all: All-frame image mask
            img_ar_mask_all: All-frame image AR mask
            train: If True, return per-sample loss and metrics

        Returns:
            (loss, metrics)
        """
        # Build prediction-specific prefix using all frames
        if self.use_gemma3:
            # Embed the prediction tokenized sequence (including placeholders)
            text_tokens_pred, text_mask_pred, text_ar_mask_pred = self._embed_text(
                observation.tokenized_prediction,
                observation.tokenized_prediction_mask,
                observation.tokenized_prediction_langact_mask,
            )
            # Replace placeholders with precomputed all-frame embeddings
            prefix_tokens_pred = self._replace_image_placeholders(
                text_tokens_pred, observation.tokenized_prediction, img_tokens_all
            )
            # Replace placeholder masks
            if text_ar_mask_pred is None:
                text_ar_mask_pred = jnp.zeros_like(text_mask_pred, dtype=bool)
            prefix_mask_pred, prefix_ar_mask_pred = self._replace_placeholder_masks(
                text_mask_pred, text_ar_mask_pred, observation.tokenized_prediction, img_mask_all, img_ar_mask_all
            )
        else:
            # Original approach: concatenate precomputed all-frame embeddings with text
            text_tokens_pred, text_mask_pred, text_ar_mask_pred = self._embed_text(
                observation.tokenized_prediction,
                observation.tokenized_prediction_mask,
                observation.tokenized_prediction_langact_mask,
            )
            prefix_tokens_pred = jnp.concatenate([img_tokens_all, text_tokens_pred], axis=1)
            prefix_mask_pred = jnp.concatenate([img_mask_all, text_mask_pred], axis=1)
            prefix_ar_mask_pred = jnp.concatenate([img_ar_mask_all, text_ar_mask_pred], axis=1)

        # Forward pass
        prefix_out_pred, _ = self._forward_language_model(prefix_tokens_pred, prefix_mask_pred, prefix_ar_mask_pred)

        # Predict next tokens over the prediction span
        shift_labels_pred = observation.tokenized_prediction[:, 1:]
        max_len_pred = observation.tokenized_prediction_langact_mask.shape[1]
        shift_tokens_pred = prefix_out_pred[:, -max_len_pred:-1, :]
        shift_logits_pred = self.PaliGemma.llm(shift_tokens_pred, method="decode")

        # Prepare token mask
        token_mask_pred = self._prepare_token_mask(
            observation.tokenized_prediction_langact_mask[:, 1:],
            observation.tokenized_prediction_mask[:, 1:],
            observation.sample_mask,
        )

        # Prepare additional masks for accuracy computation
        ex_mask_pred = jnp.asarray(observation.sample_mask)[..., None] if observation.sample_mask is not None else None
        critical_pred_mask = None
        if (
            hasattr(observation, "prediction_crictical_token_mask")
            and observation.prediction_crictical_token_mask is not None
        ):
            critical_pred_mask = (
                observation.prediction_crictical_token_mask[:, 1:] * ex_mask_pred
                if ex_mask_pred is not None
                else observation.prediction_crictical_token_mask[:, 1:]
            )

        number_pred_mask = None
        if (
            hasattr(observation, "prediction_number_token_mask")
            and observation.prediction_number_token_mask is not None
        ):
            number_pred_mask = (
                observation.prediction_number_token_mask[:, 1:] * ex_mask_pred
                if ex_mask_pred is not None
                else observation.prediction_number_token_mask[:, 1:]
            )

        direction_pred_mask = None
        if (
            hasattr(observation, "prediction_direction_token_mask")
            and observation.prediction_direction_token_mask is not None
        ):
            direction_pred_mask = (
                observation.prediction_direction_token_mask[:, 1:] * ex_mask_pred
                if ex_mask_pred is not None
                else observation.prediction_direction_token_mask[:, 1:]
            )

        # Compute loss and metrics
        pred_loss, pred_metrics = self._compute_cross_entropy_with_metrics(
            logits=shift_logits_pred,
            labels=shift_labels_pred,
            token_mask=token_mask_pred,
            critical_mask=critical_pred_mask,
            number_mask=number_pred_mask,
            direction_mask=direction_pred_mask,
            train=train,
        )

        # Rename metrics to add "pred_" prefix
        metrics = {}
        for key, value in pred_metrics.items():
            if key == "token_accuracy":
                metrics["pred_token_accuracy"] = value
            elif key == "critical_token_accuracy":
                metrics["pred_critical_token_accuracy"] = value
            elif key == "number_token_accuracy":
                metrics["pred_number_token_accuracy"] = value
            elif key == "direction_token_accuracy":
                metrics["pred_direction_token_accuracy"] = value
            else:
                metrics[key] = value

        # Store loss
        if train:
            metrics["pred_loss"] = pred_loss
        else:
            metrics["pred_loss"] = jnp.mean(pred_loss)

        return pred_loss, metrics

    def _compute_action_loss(
        self,
        observation: CoTObservation | Observation,
        actions: _model.Actions,
        prefix_tokens: at.Float[at.Array, "b s emb"],
        prefix_mask: at.Bool[at.Array, "b s"],
        noise_rng: at.KeyArrayLike,
        time_rng: at.KeyArrayLike,
        train: bool,
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        """Compute action diffusion loss.

        Args:
            observation: Observation containing state
            actions: Ground truth actions
            prefix_tokens: Prefix embeddings (images + text)
            prefix_mask: Prefix attention mask
            noise_rng: RNG for noise sampling
            time_rng: RNG for time sampling
            train: If True, return per-sample loss

        Returns:
            (loss, metrics)
        """
        # For action training, text tokens should not use autoregressive masking
        prefix_ar_mask_action = jnp.zeros_like(prefix_mask, dtype=bool)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        suffix_ar_mask = einops.repeat(suffix_ar_mask, "s -> b s", b=suffix_tokens.shape[0])

        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask_action, suffix_ar_mask], axis=1)
        attn_mask = _pi0.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        action_loss = jnp.mean(jnp.square(v_t - u_t), axis=(-1, -2))

        # Store loss in metrics
        metrics = {}
        if train:
            metrics["action_loss"] = action_loss
        else:
            metrics["action_loss"] = jnp.mean(action_loss)

        return action_loss, metrics

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: CoTObservation | Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        stage_config: dict | None = None,
    ) -> tuple[
        at.Float[at.Array, "*b ah"],
        at.Float[at.Array, ""],
        at.Float[at.Array, ""],
        at.Float[at.Array, ""],
        at.Float[at.Array, ""],
        dict[str, at.Array],
    ]:
        preprocess_rng, stochastic_rng, noise_rng, time_rng = jax.random.split(rng, 4)

        # Preprocess observation
        observation = preprocess_observation(
            preprocess_rng, observation, train=train, image_keys=self.image_keys, aug_wrist_image=self.aug_wrist_image
        )

        # Determine loss configuration from stage_config or model defaults
        if stage_config is not None:
            langact_rng, action_rng, prediction_rng = jax.random.split(stochastic_rng, 3)

            # Stochastic loss selection with probability-based enabling
            # Use JAX operations to avoid traced boolean conversion errors
            langact_enabled = stage_config.get("enable_langact_training", self.enable_langact_training)
            langact_prob = stage_config.get("langact_prob", 1.0)
            # Apply stochastic masking: if prob < 1.0, use random sampling
            langact_enabled = jnp.where(
                langact_prob < 1.0, langact_enabled & (jax.random.uniform(langact_rng) < langact_prob), langact_enabled
            )

            action_enabled = stage_config.get("enable_action_training", self.enable_action_training)
            action_prob = stage_config.get("action_prob", 1.0)
            action_enabled = jnp.where(
                action_prob < 1.0, action_enabled & (jax.random.uniform(action_rng) < action_prob), action_enabled
            )

            prediction_enabled = stage_config.get("enable_prediction_training", self.enable_prediction_training)
            prediction_prob = stage_config.get("prediction_prob", 1.0)
            prediction_enabled = jnp.where(
                prediction_prob < 1.0,
                prediction_enabled & (jax.random.uniform(prediction_rng) < prediction_prob),
                prediction_enabled,
            )

            # Get loss weights
            language_loss_weight = stage_config.get("language_loss_weight", self.language_loss_weight)
            action_loss_weight = stage_config.get("action_loss_weight", self.action_loss_weight)
            prediction_loss_weight = stage_config.get("prediction_loss_weight", self.prediction_loss_weight)
        else:
            # Use model's static configuration
            langact_enabled = self.enable_langact_training
            action_enabled = self.enable_action_training
            prediction_enabled = self.enable_prediction_training
            language_loss_weight = self.language_loss_weight
            action_loss_weight = self.action_loss_weight
            prediction_loss_weight = self.prediction_loss_weight

        # OPTIMIZATION: Encode all images once, then extract subsets for different losses
        img_tokens_all, img_mask_all, img_ar_mask_all = self._embed_images(observation, num_frames=None)
        img_tokens_first, img_mask_first, img_ar_mask_first = self._extract_first_frame_embeddings(
            observation, img_tokens_all, img_mask_all, img_ar_mask_all
        )

        # Build prefix for langact/action losses (first frame + text)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(
            observation, num_frames=1, precomputed_img_embeddings=(img_tokens_first, img_mask_first, img_ar_mask_first)
        )

        # Initialize loss accumulator and metrics
        total_loss = 0.0
        token_accuracy = jnp.array(0.0)
        critical_token_accuracy = jnp.array(0.0)
        number_token_accuracy = jnp.array(0.0)
        direction_token_accuracy = jnp.array(0.0)
        metrics = {}

        # Compute language/reasoning loss (conditionally)
        def compute_lang():
            return self._compute_language_loss(observation, prefix_tokens, prefix_mask, prefix_ar_mask, train)

        def skip_lang():
            # Return same structure with zeros - must match _compute_language_loss return type exactly
            batch_size = observation.tokenized_prompt.shape[0]
            if train:
                dummy_loss = jnp.zeros(batch_size, dtype=jnp.float32)
            else:
                dummy_loss = jnp.array(0.0, dtype=jnp.float32)
            dummy_metrics = {
                "lang_loss": jnp.zeros(batch_size, dtype=jnp.float32) if train else jnp.array(0.0, dtype=jnp.float32),
                "token_accuracy": jnp.array(0.0, dtype=jnp.float32),
                "critical_token_accuracy": jnp.array(0.0, dtype=jnp.float32),
                "number_token_accuracy": jnp.array(0.0, dtype=jnp.float32),
                "direction_token_accuracy": jnp.array(0.0, dtype=jnp.float32),
            }
            # Add per-sample metrics if training (must match dtypes: float32 for correct, int32 for total)
            if train:
                dummy_metrics.update(
                    {
                        "per_sample_critical_correct": jnp.zeros(batch_size, dtype=jnp.float32),
                        "per_sample_critical_total": jnp.zeros(batch_size, dtype=jnp.int32),
                        "per_sample_number_correct": jnp.zeros(batch_size, dtype=jnp.float32),
                        "per_sample_number_total": jnp.zeros(batch_size, dtype=jnp.int32),
                        "per_sample_direction_correct": jnp.zeros(batch_size, dtype=jnp.float32),
                        "per_sample_direction_total": jnp.zeros(batch_size, dtype=jnp.int32),
                    }
                )
            return dummy_loss, dummy_metrics, jnp.array(0.0, dtype=jnp.float32)

        lang_loss, lang_metrics, lang_token_accuracy = jax.lax.cond(langact_enabled, compute_lang, skip_lang)
        total_loss = total_loss + language_loss_weight * lang_loss
        metrics.update(lang_metrics)
        token_accuracy = lang_token_accuracy
        critical_token_accuracy = lang_metrics.get("critical_token_accuracy", jnp.array(0.0))
        number_token_accuracy = lang_metrics.get("number_token_accuracy", jnp.array(0.0))
        direction_token_accuracy = lang_metrics.get("direction_token_accuracy", jnp.array(0.0))

        # Compute prediction loss (conditionally, only if tokenized_prediction exists)
        if observation.tokenized_prediction is not None:

            def compute_pred():
                return self._compute_prediction_loss(observation, img_tokens_all, img_mask_all, img_ar_mask_all, train)

            def skip_pred():
                batch_size = observation.tokenized_prediction.shape[0]
                if train:
                    dummy_loss = jnp.zeros(batch_size, dtype=jnp.float32)
                else:
                    dummy_loss = jnp.array(0.0, dtype=jnp.float32)
                dummy_metrics = {
                    "pred_loss": jnp.zeros(batch_size, dtype=jnp.float32)
                    if train
                    else jnp.array(0.0, dtype=jnp.float32),
                    "pred_token_accuracy": jnp.array(0.0, dtype=jnp.float32),
                    "pred_critical_token_accuracy": jnp.array(0.0, dtype=jnp.float32),
                    "pred_number_token_accuracy": jnp.array(0.0, dtype=jnp.float32),
                    "pred_direction_token_accuracy": jnp.array(0.0, dtype=jnp.float32),
                }
                return dummy_loss, dummy_metrics

            pred_loss, pred_metrics = jax.lax.cond(prediction_enabled, compute_pred, skip_pred)
            total_loss = total_loss + prediction_loss_weight * pred_loss
            metrics.update(pred_metrics)

        # Compute action diffusion loss (conditionally)
        def compute_action():
            return self._compute_action_loss(
                observation, actions, prefix_tokens, prefix_mask, noise_rng, time_rng, train
            )

        def skip_action():
            batch_size = actions.shape[0]
            if train:
                dummy_loss = jnp.zeros(batch_size, dtype=jnp.float32)
            else:
                dummy_loss = jnp.array(0.0, dtype=jnp.float32)
            dummy_metrics = {
                "action_loss": jnp.zeros(batch_size, dtype=jnp.float32) if train else jnp.array(0.0, dtype=jnp.float32)
            }
            return dummy_loss, dummy_metrics

        action_loss, action_metrics = jax.lax.cond(action_enabled, compute_action, skip_action)
        total_loss = total_loss + action_loss_weight * action_loss
        metrics.update(action_metrics)

        return (
            total_loss,
            token_accuracy,
            critical_token_accuracy,
            number_token_accuracy,
            direction_token_accuracy,
            metrics,
        )

    def _slide_window_cache(
        self,
        k_cache: at.Array,
        v_cache: at.Array,
        p_mask: at.Array,
        p_ar_mask: at.Array,
    ) -> tuple[at.Array, at.Array, at.Array, at.Array]:
        """Slide the KV cache and masks left by 1 position to free the last slot.

        Args:
            k_cache: Key cache
            v_cache: Value cache
            p_mask: Position mask
            p_ar_mask: Autoregressive mask

        Returns:
            (k_cache, v_cache, p_mask, p_ar_mask) after sliding
        """
        # Shift caches and masks left by 1 to free the last slot
        k_cache = jnp.concatenate([k_cache[:, :, 1:], jnp.zeros_like(k_cache[:, :, :1])], axis=2)
        v_cache = jnp.concatenate([v_cache[:, :, 1:], jnp.zeros_like(v_cache[:, :, :1])], axis=2)
        p_mask = jnp.concatenate([p_mask[:, 1:], jnp.zeros_like(p_mask[:, :1])], axis=1)
        p_ar_mask = jnp.concatenate([p_ar_mask[:, 1:], jnp.zeros_like(p_ar_mask[:, :1])], axis=1)

        # Maintain the scratch query column at the end
        p_mask = p_mask.at[:, -1].set(True)
        p_ar_mask = p_ar_mask.at[:, -1].set(True)

        return k_cache, v_cache, p_mask, p_ar_mask

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
            None,
            observation,
            train=False,
            image_keys=self.image_keys,
            aug_wrist_image=self.aug_wrist_image,
            # image_resolution=(896, 896),
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
        finished = curr_id == self.EOS_TOKEN

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
            k_cache, v_cache, p_mask, p_ar_mask = self._slide_window_cache(k_cache, v_cache, p_mask, p_ar_mask)

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
            finished = jnp.logical_or(finished, next_id_raw == self.EOS_TOKEN)
            eos_token = jnp.asarray(self.EOS_TOKEN, dtype=next_id_raw.dtype)
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
