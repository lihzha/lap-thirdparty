import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

# --- NEW: Import your MoE model and the Gemma 3 configs ---
from gemma3 import Gemma3MoEModel  # Assumes your model is in gemma3.py
from gemma.gm.nn import _config as gemma3_config
from gemma.gm.nn import _modules as gemma3_modules
from gemma.multimodal import vision as gemma_vision
# ---

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")

def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05

        # --- 1. UNIFIED GEMMA 3 CONFIGURATION ---
        # Create one single, multimodal config for our unified model.
        # This config defines the vision encoder, text model, and attention patterns.
        gemma3_base_config = gemma3_config.TransformerConfig(
            final_logit_softcap=None,
            num_embed=262_144,
            embed_dim=2560,
            hidden_dim=2560 * 8 // 2,
            num_heads=8,
            head_dim=256,
            num_kv_heads=4,
            use_post_attn_norm=True,
            use_post_ffw_norm=True,
            use_qk_norm=True,
            # This example uses a simple global attention pattern for all layers.
            # You can customize this as needed.
            attention_types=(gemma3_modules.AttentionType.GLOBAL,) * 18,
            query_pre_attn_norm=gemma3_config.QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM,
            attn_logits_soft_cap=None,
            sliding_window_size=None, # Not using sliding attention in this example
            transpose_gating_einsum=True,
            local_base_frequency=10_000,
            global_base_frequency=1_000_000,
            global_scale_factor=8.0,
            # CRITICAL: This makes the model multimodal.
            vision_encoder=gemma_vision.SigLiPFromPatches(),
        )

        # --- 2. INSTANTIATE THE UNIFIED MoE MODEL ---
        # We replace the separate llm and img with one model.
        # num_experts=2 corresponds to the prefix and action experts.
        self.gemma3_model = nnx_bridge.ToNNX(
            Gemma3MoEModel(config=gemma3_base_config, num_experts=2)
        )
        # Initialization will happen on the first call or via lazy_init if needed.
        
        embed_dim = gemma3_base_config.embed_dim
        
        # --- 3. ACTION/STATE PROJECTION LAYERS ---
        # These remain, but their dimensions are now tied to the unified model's embed_dim.
        self.action_in_proj = nnx.Linear(config.action_dim, embed_dim, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
            self.time_mlp_out = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, embed_dim, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * embed_dim, embed_dim, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        self.action_out_proj = nnx.Linear(embed_dim, config.action_dim, rngs=rngs)

        self.deterministic = True

    @at.typecheck
    def _prepare_prefix_inputs(
        self, obs: _model.Observation
    ) -> tuple[at.Int[at.Array, "b s"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        """Prepares raw token IDs and masks for the prefix (vision + language)."""
        # This method NO LONGER creates embeddings. It just prepares token IDs.
        # The logic for handling image placeholders inside the token stream is assumed
        # to be handled by the data loading/tokenization step.
        
        # We assume obs.tokenized_prompt now contains the full sequence with image placeholders
        tokens = obs.tokenized_prompt
        input_mask = obs.tokenized_prompt_mask
        
        # The AR mask is still needed to control attention flow.
        # This logic is simplified; the original was likely more complex.
        ar_mask = jnp.zeros(tokens.shape[1], dtype=jnp.bool_)
        
        return tokens, input_mask, ar_mask
    

    @at.typecheck
    def _prepare_suffix_inputs(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"], at.Float[at.Array, "b emb"] | None]:
        """Prepares embedded action/state tokens for the suffix."""
        # This method is a bit different. Since actions/state are continuous, we still
        # need to project them into the embedding space before the main model call.
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        
        if self.pi05:
            time_emb_processed = self.time_mlp_in(time_emb)
            time_emb_processed = nnx.swish(time_emb_processed)
            time_emb_processed = self.time_mlp_out(time_emb_processed)
            adarms_cond = nnx.swish(time_emb_processed)
            action_expert_tokens = action_tokens
        else:
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_expert_tokens = self.action_time_mlp_in(action_time_tokens)
            action_expert_tokens = nnx.swish(action_expert_tokens)
            action_expert_tokens = self.action_time_mlp_out(action_expert_tokens)
            adarms_cond = None
            
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond
    

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        # Noise and time logic remains the same
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # --- REFACTORED FORWARD PASS ---
        # 1. Prepare raw prefix inputs and embedded suffix inputs.
        prefix_tokens_ids, prefix_mask, prefix_ar_mask = self._prepare_prefix_inputs(observation)
        suffix_embeddings, suffix_mask, suffix_ar_mask, adarms_cond = self._prepare_suffix_inputs(observation, x_t, time)

        # 2. Construct masks for the full sequence.
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        # 3. A SINGLE call to our unified MoE model.
        #    The model handles vision embedding internally.
        (prefix_out, suffix_out), _ = self.gemma3_model(
            prefix_tok_ids=prefix_tokens_ids,
            suffix_embeddings=suffix_embeddings,
            images=next(iter(observation.images.values())) if observation.images else None,
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond], # Pass adarms only to the action expert
        )
        
        # 4. Project action expert output and compute loss.
        v_t = self.action_out_proj(suffix_out)
        return jnp.mean(jnp.square(v_t - u_t), axis=-1)
    

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # --- REFACTORED KV CACHING ---
        # 1. Fill KV cache with a prefix-only pass.
        prefix_tokens_ids, prefix_mask, prefix_ar_mask = self._prepare_prefix_inputs(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        
        # Pass only prefix data to the model to generate the KV cache.
        _, kv_cache = self.gemma3_model(
            prefix_tok_ids=prefix_tokens_ids,
            suffix_embeddings=None, # No suffix data in this pass
            images=next(iter(observation.images.values())) if observation.images else None,
            mask=prefix_attn_mask,
            positions=positions,
            adarms_cond=[None, None],
        )

        def step(carry):
            x_t, time = carry
            suffix_embeddings, suffix_mask, suffix_ar_mask, adarms_cond = self._prepare_suffix_inputs(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # ... (masking and position logic for the suffix remains the same) ...
            full_attn_mask = ...
            positions = ...

            # 2. In the loop, do a suffix-only pass using the pre-filled cache.
            (prefix_out, suffix_out), _ = self.gemma3_model(
                prefix_tok_ids=None, # No prefix data in this pass
                suffix_embeddings=suffix_embeddings,
                images=None, # Images are already cached
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out)
            return x_t + dt * v_t, time + dt
        
        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        # ... (while loop remains the same) ...
        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0