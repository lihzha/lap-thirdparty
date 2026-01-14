"""Gemma3-specific PiCoT implementation.

This module contains the PiCoTGemma3 class which inherits from PiCoT and overrides
only the Gemma3-specific behavior:
- Image placeholder token replacement in the tokenized sequence (vs concatenation in PiCoT)
- Different EOS token (106 vs 1) and vocab size (262144 vs 257152)
- Gemma3 LLM backbone with QK-normalization (vs softcapping in Gemma2)
- SigLIP Gemma3 vision encoder with configurable resolution and automatic pooling to 256 tokens
"""

import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import openpi.models.model as _model
from openpi.models.model import Observation
from openpi.shared import array_typing as at
from typing_extensions import override

from openpi_cot.models.backbones.gemma3 import get_config as get_gemma3_config
from openpi_cot.models.model_adapter import CoTObservation
import openpi_cot.models.pi_cot_config as _pi_cot_config
import openpi_cot.models.backbones.siglip_gemma3 as _siglip_gemma3
import openpi_cot.models.backbones.gemma3 as _gemma3
from openpi_cot.models.pi_cot import PiCoT

logger = logging.getLogger("openpi")

# Gemma3 constants
GEMMA3_VOCAB_SIZE = 262_144
GEMMA3_IMAGE_TOKEN = 262145  # Placeholder token ID for image embeddings (out-of-vocab)
GEMMA3_EOS_TOKEN = 106
GEMMA3_BEGIN_IMAGE_TOKEN = 255999
GEMMA3_END_IMAGE_TOKEN = 262144

# Gemma3 SigLIP constants
SIGLIP_PATCH_SIZE = 14  # 14x14 pixel patches
SIGLIP_NUM_PATCHES = 256  # Output tokens after pooling (16x16 grid)


class PiCoTGemma3(PiCoT):
    """Gemma3-specific PiCoT implementation.
    
    Inherits from PiCoT and overrides only Gemma3-specific behavior:
    - Image placeholders in tokenized sequence are replaced with actual embeddings
      (vs concatenating [image_tokens, text_tokens] in PiCoT)
    - Different EOS token (106 instead of 1) and vocab size (262144 instead of 257152)
    - Gemma3 LLM backbone with QK-normalization (vs softcapping in Gemma2)
    - SigLIP Gemma3 encoder with automatic pooling to 256 tokens per image
    """
    
    # Gemma3-specific token constants
    EOS_TOKEN = GEMMA3_EOS_TOKEN
    VOCAB_SIZE = GEMMA3_VOCAB_SIZE
    IMAGE_TOKEN = GEMMA3_IMAGE_TOKEN
    BEGIN_IMAGE_TOKEN = GEMMA3_BEGIN_IMAGE_TOKEN
    END_IMAGE_TOKEN = GEMMA3_END_IMAGE_TOKEN

    def __init__(self, config: _pi_cot_config.PiCoTConfig, rngs: nnx.Rngs):
        """Initialize Gemma3-specific model components.
        
        Unlike PiCoT which uses PaliGemma's SigLIP, Gemma3 uses:
        - SigLIP Gemma3 encoder with automatic pooling to 256 tokens
        - Gemma3 LLM backbone with QK-normalization
        - Image resolution is configurable (default 224x224, supports up to 896x896)
        """
        # Initialize base model (skip PiCoT's __init__ which loads wrong backbone)
        _model.BaseModel.__init__(self, config.action_dim, config.action_horizon, config.max_token_len)
        
        # Validate config
        assert "gemma3" in config.paligemma_variant, "PiCoTGemma3 requires gemma3 variant"
        assert "gemma3" in config.action_expert_variant, "gemma3 must be used for both LLM and action expert"
        
        # Store config attributes (same as PiCoT)
        self.pi05 = config.pi05
        self.verbose_mode = config.verbose_mode
        self.aug_wrist_image = config.aug_wrist_image
        self.image_keys = config.image_keys
        self.image_resolution = config.image_resolution
        self.tokenizer = None  # Can be set externally for decoding
        self.use_pan_and_scan = getattr(config, "use_pan_and_scan", False)
        
        # Loss/control knobs (same as PiCoT)
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
        
        # Compute positional embedding grid size based on image resolution
        # grid_size = image_resolution / patch_size (14)
        img_h, img_w = self.image_resolution
        posemb_h = img_h // SIGLIP_PATCH_SIZE  # e.g., 896/14=64 or 224/14=16
        posemb_w = img_w // SIGLIP_PATCH_SIZE
        
        # Initialize Gemma3-specific SigLIP with configurable resolution
        img = nnx_bridge.ToNNX(
            _siglip_gemma3.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",  # Keep 256 tokens after gemma3_pooling
                gemma3_pooling=True,
                scan=True,
                dtype_mm=config.dtype,
                posemb="learn",
                posemb_shape=(posemb_h, posemb_w),  # Dynamic based on image_resolution
            )
        )
        # Initialize with correctly sized fake image
        fake_obs = config.fake_obs()
        fake_obs_image = next(iter(fake_obs.images.values()))
        b, h, w, c = fake_obs_image.shape
        new_shape = (b, 1, img_h, img_w, c)
        fake_image_resized = jax.image.resize(fake_obs_image[:, None], new_shape, method="linear")
        img.lazy_init(fake_image_resized, train=False, rngs=rngs)

        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        
        # Initialize action expert projections (same as PiCoT)
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

    # ============ Gemma3-Specific: Image Embedding ============

    def _embed_images(
        self,
        obs: CoTObservation | Observation,
    ) -> tuple[at.Float[at.Array, "b n_patches emb"], at.Bool[at.Array, "b n_patches"]]:
        """Embed images using Gemma3-specific SigLIP encoder.
        
        SigLIP Gemma3 handles 4D input [b, h, w, c] directly (no need for time dimension).
        After pooling, outputs 256 tokens per image regardless of input resolution.
        
        Returns:
            (image_embeddings, image_mask) where:
            - image_embeddings: [b, total_patches, emb] concatenated across all images
            - image_mask: [b, total_patches] mask for valid image positions
        """
        all_tokens = []
        all_masks = []
        
        for name in obs.images:
            image = obs.images[name]
            # SigLIP Gemma3 accepts [b, h, w, c] directly - no time dimension needed
            image_tokens, _ = self.PaliGemma.img(image, train=False)
            num_patches = image_tokens.shape[1]
            
            all_tokens.append(image_tokens)
            all_masks.append(
                einops.repeat(obs.image_masks[name], "b -> b s", s=num_patches)
            )
        
        image_embeddings = jnp.concatenate(all_tokens, axis=1)
        image_mask = jnp.concatenate(all_masks, axis=1)
        
        return image_embeddings, image_mask

    # ============ Gemma3-Specific: Placeholder Replacement ============

    def _replace_placeholders(
        self,
        token_embeddings: at.Float[at.Array, "b s emb"],
        token_mask: at.Bool[at.Array, "b s"],
        token_ar_mask: at.Bool[at.Array, "b s"],
        tokenized_sequence: at.Int[at.Array, "b s"],
        image_embeddings: at.Float[at.Array, "b n_patches emb"],
        image_mask: at.Bool[at.Array, "b n_patches"],
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, "b s"],
    ]:
        """Replace IMAGE_TOKEN placeholders with actual image embeddings and masks.
        
        The tokenized sequence contains IMAGE_TOKEN (262145) at positions where
        image embeddings should be inserted. This method:
        1. Finds all placeholder positions
        2. Replaces placeholder embeddings with corresponding image embeddings
        3. Updates masks to use actual image masks at placeholder positions
        
        Args:
            token_embeddings: Text embeddings (placeholders have dummy embeddings)
            token_mask: Attention mask for tokens
            token_ar_mask: Autoregressive mask for tokens
            tokenized_sequence: Token IDs with IMAGE_TOKEN placeholders
            image_embeddings: Actual image embeddings from SigLIP
            image_mask: Mask for valid image patches
            
        Returns:
            (embeddings, mask, ar_mask) with placeholders replaced
        """
        is_placeholder = tokenized_sequence == self.IMAGE_TOKEN
        b, s, emb = token_embeddings.shape
        _, n_patches = image_mask.shape
        
        # Map each placeholder to its corresponding image embedding index
        # cumsum gives 1-indexed position, subtract 1 for 0-indexed
        placeholder_idx = jnp.cumsum(is_placeholder, axis=1) - 1
        placeholder_idx = jnp.clip(placeholder_idx, 0, n_patches - 1)
        
        # Gather image embeddings and masks at placeholder positions
        batch_idx = jnp.arange(b)[:, None]
        selected_img_embs = image_embeddings[batch_idx, placeholder_idx]
        selected_img_mask = image_mask[batch_idx, placeholder_idx]
        
        # Replace: use image values where is_placeholder, else original values
        result_embeddings = jnp.where(is_placeholder[..., None], selected_img_embs, token_embeddings)
        result_mask = jnp.where(is_placeholder, selected_img_mask, token_mask)
        # Image tokens are non-autoregressive (can attend to each other)
        result_ar_mask = jnp.where(is_placeholder, False, token_ar_mask)
        
        return result_embeddings, result_mask, result_ar_mask

    # ============ Override: embed_prefix ============

    @override
    def embed_prefix(
        self,
        obs: CoTObservation | Observation,
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, "b s"],
    ]:
        """Embed images and text for the prefix using Gemma3's placeholder approach.
        
        Unlike PiCoT which concatenates [image_tokens, text_tokens], Gemma3 uses
        placeholder tokens in the text sequence that get replaced with image embeddings.
        This keeps the sequence structure matching the tokenizer's format.
        
        Returns:
            (prefix_tokens, prefix_mask, prefix_ar_mask) with placeholders replaced
        """
        # Step 1: Embed the tokenized sequence (placeholders get dummy embeddings)
        # Replace IMAGE_TOKEN (262145, out-of-vocab) with 0 to avoid indexing errors
        tokenized_safe = jnp.where(
            obs.tokenized_prompt == self.IMAGE_TOKEN,
            0,  # Temporary placeholder - will be replaced below
            obs.tokenized_prompt
        )
        token_embeddings = self.PaliGemma.llm(tokenized_safe, method="embed")
        
        # Get text masks
        token_mask = obs.tokenized_prompt_mask
        # Attention pattern for text tokens:
        # - Prompt tokens (tokenized_langact_mask=False): ar_mask=False → bidirectional
        # - Langact tokens (tokenized_langact_mask=True): ar_mask=True → causal
        # - Image placeholders will be set to ar_mask=False in _replace_placeholders
        if obs.tokenized_langact_mask is not None:
            token_ar_mask = obs.tokenized_langact_mask
        else:
            token_ar_mask = jnp.zeros_like(token_mask, dtype=bool)
        
        # Step 2: Get image embeddings from SigLIP
        image_embeddings, image_mask = self._embed_images(obs)
        
        # Step 3: Replace placeholders with actual image embeddings and masks
        prefix_tokens, prefix_mask, prefix_ar_mask = self._replace_placeholders(
            token_embeddings,
            token_mask,
            token_ar_mask,
            obs.tokenized_prompt,
            image_embeddings,
            image_mask,
        )
        
        return prefix_tokens, prefix_mask, prefix_ar_mask

    # ============ Override: _compute_language_loss ============

    @override
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
        """Compute language loss using Gemma3's vocab size."""
        targets = jax.nn.one_hot(
            observation.tokenized_prompt[:, 1:],
            self.VOCAB_SIZE,  # Gemma3: 262144
        )

        pre_logits = prefix_pre_logits[:, :-1]
        pre_logits = pre_logits[:, -targets.shape[1]:]
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

    # ============ Override: _build_prefix_action_mask ============

    @override
    def _build_prefix_action_mask(
        self,
        prefix_mask: at.Bool[at.Array, "b s"],
        observation: CoTObservation | Observation,
    ) -> at.Bool[at.Array, "b s"]:
        """Build prefix mask for action attention, excluding langact tokens.
        
        Action tokens should attend to images + prompt, but NOT langact tokens.
        For Gemma3, images are embedded inline (via placeholders) so prefix length
        equals tokenized_prompt length - no separate image prefix adjustment needed.
        """
        if observation.tokenized_langact_mask is None:
            return prefix_mask
        # Return True for images + prompt (non-langact), False for langact
        return jnp.logical_and(prefix_mask, jnp.logical_not(observation.tokenized_langact_mask))

    # ============ Inherited from PiCoT ============
    # compute_loss, sample_actions, sample_tokens are inherited from PiCoT
    # since preprocess_observation defaults to (224, 224) image resolution
