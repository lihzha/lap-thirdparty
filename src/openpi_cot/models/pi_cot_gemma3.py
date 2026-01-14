"""Gemma3-specific PiCoT implementation.

This module contains the PiCoTGemma3 class which handles Gemma3-specific:
- Image placeholder replacement logic
- 896x896 image processing with 4x4 pooling
- Different tokenizer and EOS token handling
"""

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
from openpi.shared import array_typing as at
from typing_extensions import override

from openpi_cot.models.backbones.gemma3 import get_config as get_gemma3_config
from openpi_cot.models.model_adapter import CoTObservation
from openpi_cot.models.model_adapter import preprocess_observation
import openpi_cot.models.pi_cot_config as _pi_cot_config
import openpi_cot.models.backbones.siglip_gemma3 as _siglip_gemma3
import openpi_cot.models.backbones.gemma3 as _gemma3

logger = logging.getLogger("openpi")

# Gemma3 constants
GEMMA3_VOCAB_SIZE = 262_144
GEMMA3_IMAGE_TOKEN = 262144  # Placeholder token for image embeddings
GEMMA3_EOS_TOKEN = 106
GEMMA3_BEGIN_IMAGE_TOKEN = 255999
GEMMA3_END_IMAGE_TOKEN = 262144
GEMMA3_NEW_LINE_TOKEN = 108

# Gemma3 SigLIP constants
SIGLIP_IMAGE_SIZE = 896
SIGLIP_NUM_PATCHES = 256  # After 4x4 pooling: 4096 -> 256


class PiCoTGemma3(_pi0.Pi0):
    """Gemma3-specific PiCoT implementation.
    
    Key differences from base PiCoT:
    - Uses 896x896 images with 4x4 pooling (4096 -> 256 patches)
    - Image placeholders in tokenized sequence are replaced with actual embeddings
    - Different EOS token (106 instead of 1)
    - QK-normalization instead of softcapping
    """
    
    EOS_TOKEN = GEMMA3_EOS_TOKEN
    IMAGE_TOKEN = GEMMA3_IMAGE_TOKEN
    BEGIN_IMAGE_TOKEN = GEMMA3_BEGIN_IMAGE_TOKEN
    END_IMAGE_TOKEN = GEMMA3_END_IMAGE_TOKEN

    def __init__(self, config: _pi_cot_config.PiCoTConfig, rngs: nnx.Rngs):
        _model.BaseModel.__init__(self, config.action_dim, config.action_horizon, config.max_token_len)
        
        # Validate config
        assert "gemma3" in config.paligemma_variant, "PiCoTGemma3 requires gemma3 variant"
        assert "gemma3" in config.action_expert_variant, "gemma3 must be used for both LLM and action expert"
        
        # Store config attributes
        self.pi05 = config.pi05
        self.verbose_mode = config.verbose_mode
        self.aug_wrist_image = config.aug_wrist_image
        self.image_keys = config.image_keys
        self.image_resolution = config.image_resolution
        self.tokenizer = None  # Can be set externally for decoding
        self.use_pan_and_scan = getattr(config, "use_pan_and_scan", False)
        
        # Loss/control knobs
        self.enable_action_training = bool(config.enable_action_training)
        self.enable_langact_training = bool(config.enable_langact_training)
        self.enable_prediction_training = bool(config.enable_prediction_training)
        self.enable_vqa_training = bool(config.enable_vqa_training)
        self.language_loss_weight = float(getattr(config, "language_loss_weight", 1.0))
        self.action_loss_weight = float(getattr(config, "action_loss_weight", 1.0))
        self.prediction_loss_weight = float(getattr(config, "prediction_loss_weight", 0.2))
        self.vqa_loss_weight = float(getattr(config, "vqa_loss_weight", 0.1))
        
        # Get Gemma3 configs
        paligemma_config = get_gemma3_config(config.paligemma_variant)
        action_expert_config = get_gemma3_config(config.action_expert_variant)
        
        # Initialize Gemma3 LLM with both VLM and action expert
        llm = nnx_bridge.ToNNX(
                _gemma3.Module(
                    configs=[paligemma_config, action_expert_config],
                    embed_dtype=config.dtype,
                    adarms=config.pi05,
                    stop_action_to_vlm_grad=config.stop_action_to_vlm_grad,
                    cache_dtype=config.dtype,
                )
            )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        
        # Initialize Gemma3-specific SigLIP with 896x896 images and pooling
        img = nnx_bridge.ToNNX(
            _siglip_gemma3.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",  # Keep 256 tokens after gemma3_pooling (4096→256)
                gemma3_pooling=True,
                scan=True,
                dtype_mm=config.dtype,
                posemb="learn",
                posemb_shape=(64, 64),  # 896/14 = 64
            )
        )
        # Initialize with correctly sized fake image
        fake_obs = config.fake_obs()
        fake_obs_image = next(iter(fake_obs.images.values()))
        b, h, w, c = fake_obs_image.shape
        new_shape = (b, 1, SIGLIP_IMAGE_SIZE, SIGLIP_IMAGE_SIZE, c)
        fake_image_resized = jax.image.resize(fake_obs_image[:, None], new_shape, method="linear")
        img.lazy_init(fake_image_resized, train=False, rngs=rngs)

        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        
        # Initialize action expert projections
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(
                2 * action_expert_config.width, action_expert_config.width, rngs=rngs
            )
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        self.deterministic = True

    # ============ Image Embedding Methods ============

    def _embed_images(
        self,
        obs: CoTObservation | Observation,
        num_frames: int | None = None,
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, "b s"],
    ]:
        """Embed images using Gemma3-specific SigLIP.
        
        Handles 896x896 images with 4x4 pooling (4096 -> 256 patches per image).
        
        Returns:
            (image_tokens, image_mask, image_ar_mask)
        """
        tokens = []
        input_mask = []
        _img_ar_masks = []
        
        for name in obs.images:
            image = obs.images[name]
            
            # Gemma3 expects [b, n_crops, h, w, c] or [b, t, h, w, c]
            if image.ndim == 4:
                image = image[:, None, :, :, :]  # Add temporal/crop dimension
            b, t, h, w, c = image.shape
            
            if num_frames is not None:
                image = image[:, :num_frames]
                t = num_frames
            
            # Pass through SigLIP encoder
            # Output: [b, t*num_patches, d] where num_patches = 256 after pooling
            image_tokens, _ = self.PaliGemma.img(image, train=False)
            
            num_patches = image_tokens.shape[1]
            total_seq_len = num_patches
            
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

    # ============ Text Embedding Methods ============

    def _embed_text(
        self, tokenized_text, text_mask, text_ar_mask
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, "b s"],
    ]:
        """Embed tokenized text for Gemma3.

        Replaces IMAGE_TOKEN (262144) with 0 before embedding to avoid
        out-of-bounds errors. These positions will be replaced with actual
        image embeddings in _replace_image_placeholders.
        """
        # Replace IMAGE_TOKEN (262144) with 0 before embedding
        # Vocab size is 262144, so valid indices are 0-262143
        tokenized_text_safe = jnp.where(
            tokenized_text == self.IMAGE_TOKEN,
            0,  # Use token 0 as temporary placeholder
            tokenized_text
        )
        text_tokens = self.PaliGemma.llm(tokenized_text_safe, method="embed")
        return text_tokens, text_mask, text_ar_mask

    # ============ Image Placeholder Replacement Methods ============

    def _replace_image_placeholders(
        self,
        token_embeddings: at.Float[at.Array, "b s emb"],
        tokenized_sequence: at.Int[at.Array, "b s"],
        image_embeddings: at.Float[at.Array, "b n_img*n_patches emb"],
    ) -> at.Float[at.Array, "b s emb"]:
        """Replace placeholder tokens with actual image embeddings.

        Args:
            token_embeddings: Embeddings from tokenized sequence (includes placeholder embeddings)
            tokenized_sequence: Token IDs (includes 262144 for image placeholders)
            image_embeddings: Actual image embeddings from SigLIP [b, n_img*n_patches, emb]

        Returns:
            Updated embeddings with placeholders replaced by image embeddings
        """
        # Find placeholder positions
        is_placeholder = tokenized_sequence == self.IMAGE_TOKEN

        b, s, emb = token_embeddings.shape
        _, n_img_patches, _ = image_embeddings.shape

        # Build a mapping: cumulative sum gives the image token index for each placeholder
        placeholder_indices = jnp.cumsum(is_placeholder, axis=1) - 1  # 0-indexed
        placeholder_indices = jnp.clip(placeholder_indices, 0, n_img_patches - 1)

        # Gather image embeddings according to placeholder indices
        batch_indices = jnp.arange(b)[:, None]
        selected_image_embs = image_embeddings[batch_indices, placeholder_indices]

        # Replace: where is_placeholder, use image embedding; otherwise use token embedding
        result = jnp.where(
            is_placeholder[..., None],
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
            text_ar_mask: AR mask from tokenized sequence
            tokenized_sequence: Token IDs (includes 262144 for placeholders)
            img_mask: Actual image mask
            img_ar_mask: Actual image AR mask

        Returns:
            (updated_mask, updated_ar_mask) with placeholder positions replaced
        """
        is_placeholder = tokenized_sequence == self.IMAGE_TOKEN

        b, s = text_mask.shape
        _, n_img_patches = img_mask.shape

        placeholder_indices = jnp.cumsum(is_placeholder, axis=1) - 1
        placeholder_indices = jnp.clip(placeholder_indices, 0, n_img_patches - 1)

        batch_indices = jnp.arange(b)[:, None]
        selected_img_mask = img_mask[batch_indices, placeholder_indices]
        selected_img_ar_mask = img_ar_mask[batch_indices, placeholder_indices]

        updated_mask = jnp.where(is_placeholder, selected_img_mask, text_mask)
        updated_ar_mask = jnp.where(is_placeholder, selected_img_ar_mask, text_ar_mask)

        return updated_mask, updated_ar_mask

    # ============ Main Embedding Method ============

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
        """Embed images and text for the prefix using Gemma3's placeholder approach.

        For Gemma3, the tokenized_prompt already contains image placeholders (262144).
        We embed the full sequence, then replace placeholder embeddings with actual
        image embeddings from SigLIP.

        Args:
            obs: Observation containing images and tokenized text
            num_frames: If specified, only use first num_frames of images
            precomputed_img_embeddings: Optional pre-computed image embeddings

        Returns:
            (prefix_tokens, prefix_mask, prefix_ar_mask)
        """
        # Embed the full tokenized sequence (including placeholders)
        text_tokens, text_mask, text_ar_mask = self._embed_text(
            obs.tokenized_prompt,
            obs.tokenized_prompt_mask,
            obs.tokenized_langact_mask,
        )

        # Get actual image embeddings from SigLIP
        if precomputed_img_embeddings is not None:
            img_tokens, img_mask, img_ar_mask = precomputed_img_embeddings
        else:
            img_tokens, img_mask, img_ar_mask = self._embed_images(obs, num_frames)

        # Replace placeholder embeddings with actual image embeddings
        tokens = self._replace_image_placeholders(text_tokens, obs.tokenized_prompt, img_tokens)

        # Replace placeholder masks with actual image masks
        if text_ar_mask is None:
            text_ar_mask = jnp.zeros_like(text_mask, dtype=bool)
        input_mask, ar_mask = self._replace_placeholder_masks(
            text_mask, text_ar_mask, obs.tokenized_prompt, img_mask, img_ar_mask
        )

        return tokens, input_mask, ar_mask

    # ============ Suffix and Loss Methods ============
    # These are inherited from Pi0 or need to be implemented similar to PiCoT

    def prepare_suffix(
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
        """Compute language/reasoning loss for Gemma3."""
        targets = jax.nn.one_hot(
            observation.tokenized_prompt[:, 1:],
            GEMMA3_VOCAB_SIZE,
        )

        pre_logits = prefix_pre_logits[:, :-1]
        pre_logits = pre_logits[:, -targets.shape[1] :]
        logits = self.PaliGemma.llm(pre_logits, method="decode")

        loss_mask = jnp.logical_and(
            observation.tokenized_langact_mask[:, 1:],
            jnp.logical_and(observation.tokenized_prompt_mask[:, 1:], observation.token_loss_mask[:, 1:]),
        )
        if sample_mask is not None:
            ex_mask = jnp.asarray(sample_mask)[..., None]
            loss_mask = loss_mask * ex_mask
        else:
            ex_mask = None

        logp = jax.nn.log_softmax(logits, axis=-1)
        token_pplx = jnp.sum(targets * logp, axis=-1)
        per_sample_loss = -jnp.sum(token_pplx * loss_mask, axis=-1) / jnp.clip(jnp.sum(loss_mask, -1), 1)
        metrics = {loss_name: jnp.mean(per_sample_loss)}

        if return_predictions:
            predictions = jnp.argmax(logits, axis=-1)
            metrics["predictions"] = predictions
            metrics["labels"] = observation.tokenized_prompt[:, 1:]
            metrics["token_mask"] = loss_mask

        return per_sample_loss, metrics

    def _compute_action_loss(
        self,
        suffix_out: at.Float[at.Array, "b s emb"],
        u_t: at.Float[at.Array, "b ah ad"],
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        """Compute action diffusion loss."""
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
        # For Gemma3, prefix length equals tokenized_prompt length (no separate image prefix)
        return jnp.logical_and(prefix_mask, jnp.logical_not(observation.tokenized_langact_mask))

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
        return_augmented_images: bool = False,
    ) -> dict[str, at.Array]:
        preprocess_rng, _, noise_rng, time_rng = jax.random.split(rng, 4)
        effective_verbose_mode = verbose_mode if verbose_mode is not None else self.verbose_mode
        batch_size = observation.tokenized_prompt.shape[0]

        # Compute VQA/prediction masks
        vqa_mask = None
        if self.enable_vqa_training and hasattr(observation, "is_vqa_sample") and observation.is_vqa_sample is not None:
            vqa_mask = jnp.asarray(observation.is_vqa_sample, dtype=bool)
        pred_mask = None
        if self.enable_prediction_training and hasattr(observation, "is_prediction_sample") and observation.is_prediction_sample is not None:
            pred_mask = jnp.asarray(observation.is_prediction_sample, dtype=bool)

        # Preprocess observation
        observation = preprocess_observation(
            preprocess_rng,
            observation,
            train=train,
            image_keys=self.image_keys,
            aug_wrist_image=self.aug_wrist_image,
            vqa_mask=vqa_mask,
        )

        augmented_images = observation.images if return_augmented_images else None

        # Build prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)

        suffix_inputs = self.prepare_suffix(observation, actions, noise_rng, time_rng) if self.enable_action_training else None
        prefix_mask_action = self._build_prefix_action_mask(prefix_mask, observation) if self.enable_action_training else prefix_mask
        
        combined_mask = self._build_combined_attention_mask(
            prefix_mask,
            prefix_ar_mask,
            prefix_mask_action,
            suffix_inputs["suffix_mask"] if self.enable_action_training else None,
            suffix_inputs["suffix_ar_mask"] if self.enable_action_training else None,
        )
        combined_positions = self._build_combined_positions(
            prefix_mask, prefix_mask_action, suffix_inputs["suffix_mask"] if self.enable_action_training else None
        )

        pre_logits, _ = self.PaliGemma.llm(
            embedded=[prefix_tokens, suffix_inputs["suffix_tokens"]] if self.enable_action_training else [prefix_tokens],
            positions=combined_positions,
            mask=combined_mask,
            adarms_cond=[None, suffix_inputs["adarms_cond"]] if self.enable_action_training else [None],
        )

        metrics = {}
        lang_per_sample_loss = jnp.zeros(batch_size, dtype=jnp.float32)
        action_per_sample_loss = jnp.zeros(batch_size, dtype=jnp.float32)

        if self.enable_langact_training:
            combined_langact_mask = observation.sample_mask
            lang_loss, lang_metrics = self._compute_language_loss(
                observation,
                pre_logits[0],
                sample_mask=combined_langact_mask,
                verbose_mode=effective_verbose_mode,
            )
            metrics.update(lang_metrics)
            lang_per_sample_loss += self.language_loss_weight * lang_loss

        if self.enable_action_training:
            suffix_out = pre_logits[1]
            action_loss, action_metrics = self._compute_action_loss(suffix_out, suffix_inputs["u_t"])
            action_sample_mask = jnp.ones(batch_size, dtype=bool)
            if vqa_mask is not None:
                action_sample_mask = jnp.logical_and(action_sample_mask, jnp.logical_not(vqa_mask))
            if pred_mask is not None:
                action_sample_mask = jnp.logical_and(action_sample_mask, jnp.logical_not(pred_mask))
            action_sample_mask_f = action_sample_mask.astype(jnp.float32)
            action_per_sample_loss += self.action_loss_weight * action_loss * action_sample_mask_f
            action_metrics["action_loss"] = jnp.sum(action_loss * action_sample_mask_f) / jnp.maximum(
                jnp.sum(action_sample_mask_f), 1.0
            )
            metrics.update(action_metrics)

        total_per_sample_loss = lang_per_sample_loss + action_per_sample_loss

        if self.enable_action_training:
            action_term = jnp.sum(action_per_sample_loss) / jnp.maximum(jnp.sum(action_sample_mask_f), 1.0)
            if self.enable_langact_training:
                if observation.sample_mask is not None:
                    num_active_samples = jnp.maximum(jnp.sum(observation.sample_mask), 1.0)
                    lang_term = jnp.sum(lang_per_sample_loss) / num_active_samples
                else:
                    lang_term = jnp.mean(lang_per_sample_loss)
            else:
                lang_term = 0.0
            final_loss = lang_term + action_term
        elif self.enable_langact_training and observation.sample_mask is not None:
            num_active_samples = jnp.maximum(jnp.sum(observation.sample_mask), 1.0)
            final_loss = jnp.sum(total_per_sample_loss) / num_active_samples
        else:
            final_loss = jnp.mean(total_per_sample_loss)

        if augmented_images is not None:
            metrics["augmented_images"] = augmented_images

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
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions,
            adarms_cond=[None, None],
        )

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            gemma_out, _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            suffix_out = gemma_out[1]
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    @override
    def sample_tokens(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        max_decoding_steps: int | at.Int[at.Array, ""] = 390,
        temperature: float = 0.0,
    ) -> _model.Actions:
        observation = preprocess_observation(
            None,
            observation,
            train=False,
            image_keys=list(observation.images.keys()),
            aug_wrist_image=self.aug_wrist_image,
        )

        prefix_token_embeddings, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)

        prefix_token_embeddings, prefix_mask, prefix_attn_mask = _pi0_fast.left_to_right_align(
            prefix_token_embeddings, prefix_mask, prefix_attn_mask
        )
        prefill_size = prefix_token_embeddings.shape[1]
        prefill_len = jnp.sum(prefix_mask, axis=-1)
        prefix_start = prefill_size - prefill_len

        prefix_attn_mask = jnp.pad(prefix_attn_mask, ((0, 0), (0, 0), (0, max_decoding_steps)))
        prefix_positions = jnp.cumsum(prefix_mask, axis=-1) - 1
        pre_logits, kv_cache = self.PaliGemma.llm(
            [prefix_token_embeddings, None],
            mask=prefix_attn_mask,
            positions=prefix_positions,
            adarms_cond=[None, None],
        )

        last_logit = self.PaliGemma.llm(pre_logits[0][:, -1:], method="decode")
        output_tokens = jnp.zeros((last_logit.shape[0], max_decoding_steps), dtype=jnp.int32)

        def step(carry):
            rng, last_logit, output_tokens, cache, eos_mask, step = carry

            rng, rng_step = jax.random.split(rng)
            token = jax.lax.cond(
                temperature > 0.0,
                lambda _: jax.random.categorical(rng_step, last_logit / temperature, axis=-1).astype(jnp.int32),
                lambda _: jnp.argmax(last_logit, axis=-1).astype(jnp.int32),
                operand=None,
            )
            output_tokens = _pi0_fast.put_along_last_axis(
                output_tokens, jnp.broadcast_to(step, (token.shape[0], 1)), token
            )

            eos_token = jnp.squeeze(token, axis=-1)
            eos_mask = eos_mask | (eos_token == self.EOS_TOKEN)
            all_eos = jnp.all(eos_mask)

            token_embedding = self.PaliGemma.llm(token, method="embed")
            positions = prefill_len[:, None] + step
            mask = jnp.logical_and(
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :] >= prefix_start[:, None, None],
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :]
                < (jnp.broadcast_to(prefill_size + step + 1, (prefix_start.shape[0], 1, 1))),
            )
            last_prelogit, kv_cache = self.PaliGemma.llm(
                [token_embedding, None],
                mask=mask,
                positions=positions,
                kv_cache=cache,
                adarms_cond=[None, None],
            )
            last_logit = self.PaliGemma.llm(last_prelogit[0], method="decode")

            return rng, last_logit, output_tokens, kv_cache, eos_mask, step + 1

        def cond(carry):
            _, _, _, _, eos_mask, step = carry
            return (~jnp.all(eos_mask)) & (step < max_decoding_steps)

        _, _, output_tokens, _, _, _ = jax.lax.while_loop(
            cond, step, (rng, last_logit, output_tokens, kv_cache, jnp.zeros((last_logit.shape[0],), dtype=bool), 0)
        )
        return output_tokens
