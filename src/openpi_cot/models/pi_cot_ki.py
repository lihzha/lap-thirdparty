from functools import cache
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
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
from openpi_cot.models.gemma import get_config as get_gemma_config
from openpi_cot.models.gemma2 import get_config as get_gemma2_config
from openpi_cot.models.gemma3 import get_config as get_gemma3_config
from openpi_cot.models.label_smoothing import create_digit_smoothing_kernel
from openpi_cot.models.label_smoothing import get_digit_to_token_mapping
import openpi_cot.models.pi_cot_config as _pi_cot_config
import openpi_cot.models.siglip as _siglip_gemma3

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


class PiCoTKI(_pi0.Pi0):
    BEGIN_IMAGE_TOKEN = 255999  # only for Gemma3
    END_IMAGE_TOKEN = 262144  # only for Gemma3
    NEW_LINE_TOKEN = 108  # only for Gemma3

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
        # Backward compatibility flag used in a few places
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

        # rewrite gemma in NNX. For now, use bridge.
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
            # Standard hard target loss
            per_sample_loss = cross_entropy_loss(
                logits,
                labels,
                mask=token_mask,
                axis=-1,
                per_example=True,
            )

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

        if return_kv_cache:
            return per_sample_loss, metrics, kv_cache
        return per_sample_loss, metrics

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

    def _compute_sequence_loss(
        self,
        prefix_tokens: at.Float[at.Array, "b s emb"],
        prefix_mask: at.Bool[at.Array, "b s"],
        prefix_ar_mask: at.Bool[at.Array, "b s"],
        tokenized_sequence: at.Int[at.Array, "b s"],
        tokenized_sequence_mask: at.Bool[at.Array, "b s"],
        tokenized_langact_mask: at.Bool[at.Array, "b s"],
        critical_token_mask: at.Bool[at.Array, "b s"] | None,
        number_token_mask: at.Bool[at.Array, "b s"] | None,
        direction_token_mask: at.Bool[at.Array, "b s"] | None,
        units_number_token_mask: at.Bool[at.Array, "b s"] | None,
        digit_values: at.Int[at.Array, "b s"] | None,
        sample_mask: at.Bool[at.Array, "b"] | None,
        loss_name: str,
        metric_prefix: str,
        verbose_mode: bool = False,
        return_predictions: bool = False,
        return_kv_cache: bool = False,
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        """Shared logic for computing sequence prediction loss (language or prediction).

        Args:
            prefix_tokens: Prefix embeddings (images + text)
            prefix_mask: Prefix attention mask
            prefix_ar_mask: Prefix autoregressive mask
            tokenized_sequence: Tokenized sequence to predict
            tokenized_sequence_mask: Mask for tokenized sequence
            tokenized_langact_mask: Language/action mask for the sequence
            critical_token_mask: Mask for critical tokens
            number_token_mask: Mask for number tokens
            direction_token_mask: Mask for direction tokens
            units_number_token_mask: Mask for units digit tokens
            digit_values: Digit values (0-9) for number tokens
            sample_mask: Per-sample mask for batch-level filtering
            loss_name: Name to use for loss metric (e.g., "lang_loss", "pred_loss")
            metric_prefix: Prefix to add to metric names (e.g., "", "pred_")
            verbose_mode: Whether to compute detailed metrics
            return_predictions: Whether to return predictions and labels
            return_kv_cache: Whether to also return the KV cache from the prefix forward pass

        Returns:
            (loss, metrics)
        """
        # Forward pass
        prefix_out, kv_cache = self._forward_language_model(prefix_tokens, prefix_mask, prefix_ar_mask)

        # Predict next tokens
        shift_labels = tokenized_sequence[:, 1:]
        max_len = tokenized_langact_mask.shape[1]
        shift_tokens = prefix_out[:, -max_len:-1, :]
        shift_logits = self.PaliGemma.llm(shift_tokens, method="decode")

        # Prepare token mask
        token_mask = self._prepare_token_mask(
            tokenized_langact_mask[:, 1:],
            tokenized_sequence_mask[:, 1:],
            sample_mask,
        )

        # Prepare masks for label smoothing and accuracy metrics
        ex_mask = jnp.asarray(sample_mask)[..., None] if sample_mask is not None else None

        def prepare_mask(mask):
            if mask is None:
                return None
            shifted_mask = mask[:, 1:]
            if ex_mask is not None:
                return shifted_mask * ex_mask
            return shifted_mask

        # Always prepare units_number_mask and digit_values for label smoothing (if available)
        units_mask_shifted = prepare_mask(units_number_token_mask)
        digit_values_shifted = digit_values[:, 1:] if digit_values is not None else None

        # Only prepare detailed masks if verbose_mode is enabled (for accuracy metrics)
        if verbose_mode:
            critical_mask = prepare_mask(critical_token_mask)
            number_mask = prepare_mask(number_token_mask)
            direction_mask = prepare_mask(direction_token_mask)
        else:
            critical_mask, number_mask, direction_mask = None, None, None

        # Use pre-computed smoothing kernel if label smoothing is enabled
        smoothing_kernel = None
        if self.enable_number_label_smoothing:
            # Cast kernel to match logits dtype if needed
            smoothing_kernel = self.smoothing_kernel.astype(shift_logits.dtype)

        # Compute loss and metrics
        per_sample_loss, raw_metrics = self._compute_cross_entropy_with_metrics(
            logits=shift_logits,
            labels=shift_labels,
            token_mask=token_mask,
            critical_mask=critical_mask,
            number_mask=number_mask,
            direction_mask=direction_mask,
            units_number_mask=units_mask_shifted,
            digit_values=digit_values_shifted,
            smoothing_kernel=smoothing_kernel,
            verbose_mode=verbose_mode,
            return_predictions=return_predictions,
        )

        # Apply metric prefix if needed
        metrics = {loss_name: jnp.mean(per_sample_loss)}

        # Add predictions/labels or accuracy metrics to output
        if return_predictions or verbose_mode:
            metric_rename_map = {
                "token_accuracy": f"{metric_prefix}token_accuracy",
                "critical_token_accuracy": f"{metric_prefix}critical_token_accuracy",
                "number_token_accuracy": f"{metric_prefix}number_token_accuracy",
                "direction_token_accuracy": f"{metric_prefix}direction_token_accuracy",
            }

            for key, value in raw_metrics.items():
                # Keep predictions/labels/mask as-is without prefix
                if key in ["predictions", "labels", "token_mask"]:
                    metrics[key] = value
                # Rename accuracy metrics with prefix
                else:
                    new_key = metric_rename_map.get(key, key)
                    metrics[new_key] = value

        return per_sample_loss, metrics

    def _compute_language_loss(
        self,
        observation: CoTObservation | Observation,
        prefix_tokens: at.Float[at.Array, "b s emb"],
        prefix_mask: at.Bool[at.Array, "b s"],
        prefix_ar_mask: at.Bool[at.Array, "b s"],
        sample_mask: at.Bool[at.Array, "b"] | None = None,
        verbose_mode: bool = False,
        return_predictions: bool = False,
        return_kv_cache: bool = False,
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        """Compute language/reasoning cross-entropy loss and accuracy metrics.

        Args:
            observation: Observation containing tokenized prompts and masks
            prefix_tokens: Prefix embeddings (images + text)
            prefix_mask: Prefix attention mask
            prefix_ar_mask: Prefix autoregressive mask
            sample_mask: Optional per-sample mask to override observation.sample_mask
            verbose_mode: Whether to compute detailed metrics
            return_predictions: Whether to return predictions and labels
            return_kv_cache: Whether to also return the prefix KV cache

        Returns:
            (loss, metrics, token_accuracy_scalar)
        """
        # Use provided sample_mask or fall back to observation's sample_mask
        effective_sample_mask = sample_mask if sample_mask is not None else observation.sample_mask

        seq_outputs = self._compute_sequence_loss(
            prefix_tokens=prefix_tokens,
            prefix_mask=prefix_mask,
            prefix_ar_mask=prefix_ar_mask,
            tokenized_sequence=observation.tokenized_prompt,
            tokenized_sequence_mask=observation.tokenized_prompt_mask,
            tokenized_langact_mask=observation.tokenized_langact_mask,
            critical_token_mask=observation.critical_token_mask,
            number_token_mask=observation.number_token_mask,
            direction_token_mask=observation.direction_token_mask,
            units_number_token_mask=getattr(observation, "units_number_token_mask", None),
            digit_values=getattr(observation, "digit_values", None),
            sample_mask=effective_sample_mask,
            loss_name="lang_loss",
            metric_prefix="",
            verbose_mode=verbose_mode,
            return_predictions=return_predictions,
            return_kv_cache=return_kv_cache,
        )

        if return_kv_cache:
            per_sample_loss, metrics, kv_cache = seq_outputs
            return per_sample_loss, metrics, kv_cache
        return seq_outputs

    def _compute_action_loss(
        self,
        observation: CoTObservation | Observation,
        actions: _model.Actions,
        prefix_tokens: at.Float[at.Array, "b s emb"],
        prefix_mask: at.Bool[at.Array, "b s"],
        prefix_kv_cache,
        noise_rng: at.KeyArrayLike,
        time_rng: at.KeyArrayLike,
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        """Compute action diffusion loss.

        Args:
            observation: Observation containing state
            actions: Ground truth actions
            prefix_tokens: Prefix embeddings (images + text)
            prefix_mask: Prefix attention mask
            prefix_kv_cache: Cached KV tensors from the prefix forward pass
            noise_rng: RNG for noise sampling
            time_rng: RNG for time sampling

        Returns:
            (loss, metrics)
        """
        # For action training, text tokens should not use autoregressive masking
        prefix_ar_mask_action = jnp.zeros_like(prefix_mask, dtype=bool)

        # Mask out langact tokens so actions cannot see them
        if observation.tokenized_langact_mask is not None:
            if self.use_gemma3:
                # For Gemma3, prefix corresponds directly to tokenized_prompt
                # Mask out positions where langact tokens are present
                prefix_mask_action = jnp.logical_and(prefix_mask, jnp.logical_not(observation.tokenized_langact_mask))
            else:
                # For legacy, prefix = [img_tokens, text_tokens]
                # Compute image sequence length
                img_seq_len = prefix_mask.shape[1] - observation.tokenized_langact_mask.shape[1]
                # Pad langact mask with False for image positions
                langact_mask_full = jnp.concatenate(
                    [
                        jnp.zeros((observation.tokenized_langact_mask.shape[0], img_seq_len), dtype=bool),
                        observation.tokenized_langact_mask,
                    ],
                    axis=1,
                )
                # Mask out langact positions
                prefix_mask_action = jnp.logical_and(prefix_mask, jnp.logical_not(langact_mask_full))
        else:
            prefix_mask_action = prefix_mask

        if prefix_kv_cache is None:
            # Build the cache using the utility helper so the action expert respects
            # its own masking when language training is disabled.
            _, prefix_kv_cache = self._forward_language_model(prefix_tokens, prefix_mask_action, prefix_ar_mask_action)
        prefix_kv_cache = jtu.tree_map(jax.lax.stop_gradient, prefix_kv_cache)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        suffix_ar_mask = einops.repeat(suffix_ar_mask, "s -> b s", b=suffix_tokens.shape[0])

        suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_to_suffix_mask = einops.repeat(prefix_mask_action, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_to_suffix_mask, suffix_attn_mask], axis=-1)
        suffix_positions = jnp.sum(prefix_mask_action, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=suffix_positions,
            kv_cache=prefix_kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        per_sample_action_loss = jnp.mean(jnp.square(v_t - u_t), axis=(-1, -2))

        action_loss = jnp.mean(per_sample_action_loss)

        # Store loss in metrics
        metrics = {}
        metrics["action_loss"] = action_loss

        return per_sample_action_loss, metrics

    def _compute_sample_specific_metrics(
        self,
        per_sample_loss: at.Float[at.Array, "b"],
        lang_metrics: dict[str, at.Array],
        sample_mask: at.Bool[at.Array, "b"],
        prefix: str,
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
        preprocess_rng, stochastic_rng, noise_rng, time_rng = jax.random.split(rng, 4)

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

        # if stage_config is not None:
        #     langact_rng, action_rng = jax.random.split(stochastic_rng, 2)

        #     # Per-sample stochastic masking using probabilities
        #     langact_enabled = stage_config.get("enable_langact_training", self.enable_langact_training)
        #     langact_prob = stage_config.get("langact_prob", 1.0)
        #     # Create per-sample mask: True for samples that should compute this loss
        #     langact_sample_mask = jax.random.uniform(langact_rng, (batch_size,)) < langact_prob
        #     # Combine with enable flag
        #     langact_sample_mask = jnp.where(langact_enabled, langact_sample_mask, jnp.zeros(batch_size, dtype=bool))

        #     action_enabled = stage_config.get("enable_action_training", self.enable_action_training)
        #     action_prob = stage_config.get("action_prob", 1.0)
        #     action_sample_mask = jax.random.uniform(action_rng, (batch_size,)) < action_prob
        #     action_sample_mask = jnp.where(action_enabled, action_sample_mask, jnp.zeros(batch_size, dtype=bool))

        #     # Get loss weights
        #     language_loss_weight = stage_config.get("language_loss_weight", self.language_loss_weight)
        #     action_loss_weight = stage_config.get("action_loss_weight", self.action_loss_weight)
        #     prediction_loss_weight = stage_config.get("prediction_loss_weight", self.prediction_loss_weight)
        #     vqa_loss_weight = stage_config.get("vqa_loss_weight", self.vqa_loss_weight)
        # else:
        # # Use model's static configuration
        # langact_enabled = self.enable_langact_training
        # action_enabled = self.enable_action_training

        # # Create full masks when no stage_config (all samples included if enabled)
        # langact_sample_mask = jnp.full(batch_size, langact_enabled, dtype=bool)
        # action_sample_mask = jnp.full(batch_size, action_enabled, dtype=bool)

        # language_loss_weight = self.language_loss_weight
        # action_loss_weight = self.action_loss_weight
        # prediction_loss_weight = self.prediction_loss_weight
        # vqa_loss_weight = self.vqa_loss_weight

        # Encode images (only first frame needed since prediction is handled at dataset level)
        img_tokens_first, img_mask_first, img_ar_mask_first = self._embed_images(observation, num_frames=1)

        # Build prefix for langact/action losses (first frame + text)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(
            observation, num_frames=1, precomputed_img_embeddings=(img_tokens_first, img_mask_first, img_ar_mask_first)
        )
        prefix_kv_cache = None
        need_prefix_cache = self.enable_action_training

        # Initialize loss accumulator and metrics
        metrics = {}
        total_per_sample_loss = 0

        # Compute language/reasoning loss with per-sample masking
        # Combine langact_sample_mask with existing sample_mask
        if self.enable_langact_training:
            combined_langact_mask = observation.sample_mask
            # combined_langact_mask = langact_sample_mask
            # if observation.sample_mask is not None:
            #     combined_langact_mask = jnp.logical_and(combined_langact_mask, observation.sample_mask)

            # Pass combined mask to language loss computation
            lang_outputs = self._compute_language_loss(
                observation,
                prefix_tokens,
                prefix_mask,
                prefix_ar_mask,
                sample_mask=combined_langact_mask,
                verbose_mode=effective_verbose_mode,
                return_kv_cache=need_prefix_cache,
            )
            if need_prefix_cache:
                lang_loss, lang_metrics, prefix_kv_cache = lang_outputs
            else:
                lang_loss, lang_metrics = lang_outputs

            if self.enable_vqa_training or self.enable_prediction_training:
                # Create masks for each sample type
                # VQA mask was already computed at the top
                if vqa_mask is None:
                    vqa_mask = jnp.zeros(batch_size, dtype=bool)

                pred_mask = (
                    jnp.asarray(observation.is_prediction_sample, dtype=bool)
                    if self.enable_prediction_training
                    else jnp.zeros(batch_size, dtype=bool)
                )
                lang_mask = jnp.logical_not(jnp.logical_or(vqa_mask, pred_mask))

                # Combine with langact mask to get final masks
                vqa_mask = jnp.logical_and(vqa_mask, combined_langact_mask)
                pred_mask = jnp.logical_and(pred_mask, combined_langact_mask)
                lang_mask = jnp.logical_and(lang_mask, combined_langact_mask)

                # Compute comprehensive metrics for VQA samples
                if self.enable_vqa_training:
                    vqa_metrics = self._compute_sample_specific_metrics(
                        per_sample_loss=lang_loss,
                        lang_metrics=lang_metrics,
                        sample_mask=vqa_mask,
                        prefix="vqa_",
                        verbose_mode=effective_verbose_mode,
                    )
                    metrics.update(vqa_metrics)

                # Compute comprehensive metrics for prediction samples
                if self.enable_prediction_training:
                    pred_metrics = self._compute_sample_specific_metrics(
                        per_sample_loss=lang_loss,
                        lang_metrics=lang_metrics,
                        sample_mask=pred_mask,
                        prefix="pred_",
                        verbose_mode=effective_verbose_mode,
                    )
                    metrics.update(pred_metrics)

                # Compute comprehensive metrics for language-action samples
                langact_metrics = self._compute_sample_specific_metrics(
                    per_sample_loss=lang_loss,
                    lang_metrics=lang_metrics,
                    sample_mask=lang_mask,
                    prefix="langact_",
                    verbose_mode=effective_verbose_mode,
                )
                metrics.update(langact_metrics)

                total_per_sample_loss += (
                    self.vqa_loss_weight * lang_loss * vqa_mask
                    + self.prediction_loss_weight * lang_loss * pred_mask
                    + self.language_loss_weight * lang_loss * lang_mask
                )
            else:
                # No VQA or prediction masks available, use original behavior
                langact_metrics = self._compute_sample_specific_metrics(
                    per_sample_loss=lang_loss,
                    lang_metrics=lang_metrics,
                    sample_mask=combined_langact_mask,
                    prefix="langact_",
                    verbose_mode=effective_verbose_mode,
                )
                metrics.update(langact_metrics)
                total_per_sample_loss += self.language_loss_weight * lang_loss

        # Compute action diffusion loss only if action training is enabled
        if self.enable_action_training:
            action_loss, action_metrics = self._compute_action_loss(
                observation, actions, prefix_tokens, prefix_mask, prefix_kv_cache, noise_rng, time_rng
            )
            # Apply loss only to masked samples
            # total_per_sample_loss += self.action_loss_weight * action_loss * action_sample_mask
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
    def compute_loss_with_decoded_tokens(
        self,
        rng: at.KeyArrayLike,
        observation: CoTObservation | Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        verbose_mode: bool = False,
    ) -> tuple[float, dict[str, at.Array]]:
        """Compute loss and return predictions and labels for visualization.

        This is similar to compute_loss but optimized for visualization:
        - Always returns predicted tokens, ground truth labels, and token masks
        - By default, skips expensive accuracy metric computation (verbose_mode=False)
        - Avoids preparing critical/number/direction masks when not needed

        Args:
            rng: Random key
            observation: Observation containing images and tokenized text
            actions: Ground truth actions (unused but kept for API compatibility)
            train: Whether in training mode
            verbose_mode: Whether to compute detailed accuracy metrics (default: False for efficiency)

        Returns:
            (loss, metrics) where metrics includes 'predictions', 'labels', and 'token_mask'
        """
        preprocess_rng, _, _, _ = jax.random.split(rng, 4)

        # Use explicit verbose_mode (defaults to False for efficiency)
        effective_verbose_mode = verbose_mode

        # Preprocess observation
        vqa_mask = None
        if self.enable_vqa_training and hasattr(observation, "is_vqa_sample") and observation.is_vqa_sample is not None:
            vqa_mask = jnp.asarray(observation.is_vqa_sample, dtype=bool)

        observation = preprocess_observation(
            preprocess_rng,
            observation,
            train=train,
            image_keys=self.image_keys,
            aug_wrist_image=self.aug_wrist_image,
            vqa_mask=vqa_mask,
        )

        # Encode images (only first frame needed)
        img_tokens_first, img_mask_first, img_ar_mask_first = self._embed_images(observation, num_frames=1)

        # Build prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(
            observation, num_frames=1, precomputed_img_embeddings=(img_tokens_first, img_mask_first, img_ar_mask_first)
        )

        # Compute language loss with predictions
        # NOTE: We always set return_predictions=True because this method's purpose
        # is to return decoded tokens for visualization
        combined_langact_mask = observation.sample_mask
        lang_loss, lang_metrics = self._compute_language_loss(
            observation,
            prefix_tokens,
            prefix_mask,
            prefix_ar_mask,
            sample_mask=combined_langact_mask,
            verbose_mode=effective_verbose_mode,
            return_predictions=True,  # Always True: this method is for visualization
        )

        # Extract predictions, labels, and mask from metrics
        # These are always present because return_predictions=True above

        # Compute loss with correct normalization (same as compute_loss)
        if observation.sample_mask is not None:
            num_active_samples = jnp.maximum(jnp.sum(observation.sample_mask), 1.0)
            final_loss = jnp.sum(lang_loss) / num_active_samples
        else:
            final_loss = jnp.mean(lang_loss)

        metrics = {
            "loss": final_loss,
            "predictions": lang_metrics["predictions"],
            "labels": lang_metrics["labels"],
            "token_mask": lang_metrics["token_mask"],
        }

        # Include accuracy metrics if verbose mode is enabled
        # (predictions/labels/token_mask are already included above)
        if effective_verbose_mode:
            for key, value in lang_metrics.items():
                if key not in ["predictions", "labels", "token_mask"]:
                    metrics[key] = value

        return final_loss, metrics

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
