# pi0_gemini.py (Final Fix)

import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

import openpi_cot.models.gemini.gemma3_gemini as _gemma3
from openpi_cot.models.gemini import pi0_config_gemini as pi0_config
from openpi.models import model as _model
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


# (Helper functions make_attn_mask and posemb_sincos are unchanged)
def make_attn_mask(input_mask, mask_ar):
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)

@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")
    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum("i,j->ij", pos, 1.0 / period * 2 * jnp.pi, precision=jax.lax.Precision.HIGHEST)
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0Gemma3(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.config = config

        gemma_config = config.get_gemma3_config()
        self.Gemma3 = nnx_bridge.ToNNX(
            _gemma3.Gemma3MoEModel(config=gemma_config, num_experts=2)
        )

        # --- THIS IS THE FIX ---

        # 1. Create ALL the dummy data needed for the full trace.
        obs_spec, _ = config.inputs_spec(batch_size=1)
        action_expert_width = gemma_config.embed_dim
        
        # Dummy data for `embed_prefix`
        dummy_tokens = jnp.zeros(obs_spec.tokenized_prompt.shape, dtype=obs_spec.tokenized_prompt.dtype)
        first_image_spec = next(iter(obs_spec.images.values()))
        dummy_image = jnp.zeros(first_image_spec.shape, dtype=first_image_spec.dtype)
        
        # Dummy data for `__call__`
        dummy_prefix_emb = jnp.zeros((1, config.max_token_len, action_expert_width))
        dummy_suffix_emb = jnp.zeros((1, config.action_horizon + 1, action_expert_width))
        total_seq_len = dummy_prefix_emb.shape[1] + dummy_suffix_emb.shape[1]
        dummy_positions = jnp.arange(total_seq_len)[None, :]
        dummy_mask = jnp.ones((1, 1, total_seq_len, total_seq_len), dtype=bool)

        # --- THIS IS THE INITIALIZATION FIX ---
        # Create a dummy condition for the action expert (expert 1)
        dummy_cond = jnp.zeros((1, action_expert_width))
        dummy_adarms_cond_list = [None, dummy_cond] # Expert 0 has no condition

        self.Gemma3.lazy_init(
            rngs=rngs,
            method="init_all",
            dummy_images=dummy_image,
            dummy_tokens=dummy_tokens,
            dummy_embedded_inputs=[dummy_prefix_emb, dummy_suffix_emb],
            dummy_positions=dummy_positions,
            dummy_mask=dummy_mask,
            dummy_adarms_cond=dummy_adarms_cond_list, # Pass the dummy condition
        )

        # --- End of Fix ---

        # # --- THIS IS THE FIX ---
        # # 1. Get the abstract shapes and dtypes from the config spec.
        # obs_spec, _ = config.inputs_spec(batch_size=1)
        # action_expert_width = gemma_config.embed_dim

        # dummy_tokens = jnp.zeros(obs_spec.tokenized_prompt.shape, dtype=obs_spec.tokenized_prompt.dtype)
        # first_image_spec = next(iter(obs_spec.images.values()))
        # dummy_image = jnp.zeros(first_image_spec.shape, dtype=first_image_spec.dtype)
        
        # # Initialize the `embed_prefix` method (this part is correct)
        # self.Gemma3.lazy_init(
        #     rngs=rngs,
        #     method="embed_prefix",
        #     images=dummy_image,
        #     tokens=dummy_tokens,
        # )

        # # --- THIS IS THE FIX ---
        # # 1. Create dummy embeddings to determine the total sequence length.
        # dummy_prefix_emb = jnp.zeros((1, config.max_token_len, action_expert_width))
        # dummy_suffix_emb = jnp.zeros((1, config.action_horizon + 1, action_expert_width))
        # total_seq_len = dummy_prefix_emb.shape[1] + dummy_suffix_emb.shape[1]

        # # 2. Create `positions` and `mask` with shapes consistent with this length.
        # dummy_positions = jnp.arange(total_seq_len)[None, :] # Shape [1, total_seq_len]
        # # Mask shape should be [B, 1, Q_len, K_len]. For init, a simple broadcastable mask is fine.
        # dummy_mask = jnp.ones((1, 1, total_seq_len, total_seq_len), dtype=bool)

        # # 3. Initialize the `__call__` method using these correctly shaped dummy tensors.
        # self.Gemma3.lazy_init(
        #     rngs=rngs,
        #     method="__call__",
        #     embedded_inputs=[dummy_prefix_emb, dummy_suffix_emb],
        #     positions=dummy_positions,
        #     mask=dummy_mask,
        # )

        # --- End of Fix ---

        # --- 4. Define External Projection Layers (Unchanged) ---
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_width, action_expert_width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_width, action_expert_width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_width, action_expert_width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_width, action_expert_width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_width, config.action_dim, rngs=rngs)
        self.deterministic = True
        
    # ... The rest of the file (compute_loss, sample_actions, etc.) is correct and does not need to change.
    def _prepare_suffix_embeddings(self, noisy_actions, timestep):
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        action_tokens = self.action_in_proj(noisy_actions)
        adarms_cond = None
        if self.config.pi05:
            time_emb_mlp = self.time_mlp_in(time_emb)
            time_emb_mlp = nnx.swish(time_emb_mlp)
            time_emb_mlp = self.time_mlp_out(time_emb_mlp)
            adarms_cond = nnx.swish(time_emb_mlp)
            suffix_embedded = action_tokens
        else:
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            suffix_embedded = self.action_time_mlp_out(action_time_tokens)
        return suffix_embedded, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = (1 - time_expanded) * actions + time_expanded * noise
        u_t = noise - actions
        first_image = next(iter(observation.images.values()))
        prefix_embedded = self.Gemma3(
            images=first_image,
            tokens=observation.tokenized_prompt,
            method="embed_prefix"  # Tell the wrapper which method to run
        )
        print(f"DEBUG: prefix_embedded shape: {prefix_embedded.shape}")
        suffix_embedded, adarms_cond = self._prepare_suffix_embeddings(x_t, time)
        if not self.config.pi05:
            state_token = self.state_proj(observation.state)[:, None, :]
            suffix_embedded = jnp.concatenate([state_token, suffix_embedded], axis=1)

        # 1. Create masks and positions as before.
        prefix_mask = jnp.ones(prefix_embedded.shape[:2], dtype=bool)
        suffix_mask = jnp.ones(suffix_embedded.shape[:2], dtype=bool)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        # (ar_mask logic is correct)
        # NOTE: ar_mask was missing from your snippet, re-adding it here.
        ar_mask_prefix = jnp.zeros(prefix_mask.shape[1], dtype=bool)
        ar_mask_suffix = [True] * suffix_mask.shape[1]
        if not self.config.pi05:
            ar_mask_suffix[0] = False
        ar_mask = jnp.array(list(ar_mask_prefix) + ar_mask_suffix)
        
        attn_mask_3d = make_attn_mask(input_mask, ar_mask) # Shape [B, L, L]
        positions_1d = jnp.arange(input_mask.shape[1])    # Shape [L]

        # 2. Add the necessary dimensions for the Gemma 3 attention module.
        batch_size = prefix_embedded.shape[0]
        final_attn_mask = attn_mask_3d[:, None, :, :]          # Shape [B, 1, L, L]
        final_positions = einops.repeat(positions_1d, "l -> b l", b=batch_size) # Shape [B, L]
        
        # --- End of Fix ---
        
        _, suffix_out = self.Gemma3(
            embedded_inputs=[prefix_embedded, suffix_embedded],
            positions=final_positions, # Pass correctly shaped positions
            mask=final_attn_mask,      # Pass correctly shaped mask
            adarms_cond=[None, adarms_cond]
        )[0]
        
        # --- End of Fix ---
        # prefix_mask = jnp.ones(prefix_embedded.shape[:2], dtype=bool)
        # suffix_mask = jnp.ones(suffix_embedded.shape[:2], dtype=bool)
        # input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        # ar_mask_prefix = jnp.zeros(prefix_mask.shape[1], dtype=bool)
        # ar_mask_suffix = [True] * suffix_mask.shape[1]
        # if not self.config.pi05:
        #      ar_mask_suffix[0] = False
        # ar_mask = jnp.array(list(ar_mask_prefix) + ar_mask_suffix)
        # attn_mask = make_attn_mask(input_mask, ar_mask)
        # positions = jnp.arange(input_mask.shape[1])
        # _, suffix_out = self.Gemma3(
        #     embedded_inputs=[prefix_embedded, suffix_embedded],
        #     positions=positions,
        #     mask=attn_mask,
        #     adarms_cond=[None, adarms_cond]
        # )[0]
        v_t = self.action_out_proj(suffix_out)
        if not self.config.pi05:
            v_t = v_t[:, 1:, :]
        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self, rng: at.KeyArrayLike, observation: _model.Observation, *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        first_image = next(iter(observation.images.values()))
        prefix_embedded = self.Gemma3(
            images=first_image,
            tokens=observation.tokenized_prompt,
            method="embed_prefix"  # Tell the wrapper which method to run
        )
        # Check shape of prefix_embedded
        #print(f"DEBUG: prefix_embedded shape: {prefix_embedded.shape}")
        
        prefix_mask = jnp.ones(prefix_embedded.shape[:2], dtype=bool)
        prefix_ar_mask = jnp.zeros(prefix_mask.shape[1], dtype=bool)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_attn_mask = prefix_attn_mask[:, None, :, :] # Add the head dimension
        prefix_positions = jnp.arange(prefix_embedded.shape[1])
        _, kv_cache = self.Gemma3(
            embedded_inputs=[prefix_embedded, None],
            positions=prefix_positions,
            mask=prefix_attn_mask,
        )
        
        def step_fn(carry):
            x_t, time = carry
            # (embedding creation is correct)
            suffix_embedded, adarms_cond = self._prepare_suffix_embeddings(x_t, jnp.full(batch_size, time))
            if not self.config.pi05:
                state_token = self.state_proj(observation.state)[:, None, :]
                suffix_embedded = jnp.concatenate([state_token, suffix_embedded], axis=1)
            
            # Create correctly shaped mask for the suffix attending to the prefix + suffix.
            suffix_mask = jnp.ones(suffix_embedded.shape[:2], dtype=bool)
            full_input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
            
            # The AR mask should cover the full sequence.
            ar_mask_suffix = [True] * suffix_mask.shape[1]
            if not self.config.pi05:
                ar_mask_suffix[0] = False
            full_ar_mask = jnp.array(list(prefix_ar_mask) + ar_mask_suffix)
            
            # Create a full square mask first.
            full_attn_mask_3d_square = make_attn_mask(full_input_mask, full_ar_mask)

            # Now, slice out only the rows corresponding to the suffix queries.
            # This creates the correct [B, Suffix_Len, Full_Len] shape.
            asymmetric_attn_mask_3d = full_attn_mask_3d_square[:, prefix_mask.shape[1]:, :]
            
            # Add the head dimension.
            final_attn_mask = asymmetric_attn_mask_3d[:, None, :, :]
            # --- End of Fix ---
            
            suffix_positions_1d = prefix_embedded.shape[1] + jnp.arange(suffix_embedded.shape[1])
            suffix_positions = einops.repeat(suffix_positions_1d, "l -> b l", b=batch_size)
            
            # --- End of Fix ---

            _, suffix_out = self.Gemma3(
                embedded_inputs=[None, suffix_embedded],
                positions=suffix_positions,
                mask=final_attn_mask,
                adarms_cond=[None, adarms_cond],
                kv_cache=kv_cache,
            )[0]
            
            v_t = self.action_out_proj(suffix_out)
            if not self.config.pi05:
                v_t = v_t[:, 1:, :]
            x_t_next = x_t + dt * v_t
            time_next = time + dt
            return x_t_next, time_next
        def cond_fn(carry):
            _, time = carry
            return time >= -dt / 2
        x_0, _ = jax.lax.while_loop(cond_fn, step_fn, (noise, 1.0))
        return x_0