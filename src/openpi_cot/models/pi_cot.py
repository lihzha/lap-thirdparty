import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import openpi.models.gemma as _gemma
import openpi.models.model as _model
import openpi.models.pi0 as _pi0
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
from typing_extensions import override

from openpi_cot.models.adapters.gemma_adapter import ModuleWithDecode
from openpi_cot.models.adapters.model_adapter import CoTObservation
from openpi_cot.models.adapters.model_adapter import preprocess_observation
import openpi_cot.models.pi_cot_config as _pi_cot_config

logger = logging.getLogger("openpi")


def put_along_last_axis(arr, indices, values):
    """Like np.put_along_axis(..., axis=-1), since jax is missing it."""
    assert arr.ndim == indices.ndim == values.ndim, (arr.ndim, indices.ndim, values.ndim)
    onehot = jax.nn.one_hot(indices, arr.shape[-1], dtype=values.dtype)
    put_mask = jnp.einsum("...i,...in->...n", jnp.ones(values.shape, jnp.int32), onehot)
    put_values = jnp.einsum("...i,...in->...n", values, onehot)
    return jnp.where(put_mask, put_values, arr)


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
    lang_action_only: bool = True

    def __init__(self, config: _pi_cot_config.PiCoTConfig, rngs: nnx.Rngs):
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            ModuleWithDecode(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
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
        self.number_token_weight = float(config.number_token_weight)

    @at.typecheck
    def embed_prefix(
        self, obs: CoTObservation
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, "b s"],
    ]:
        input_mask = []
        tokens = []
        _img_ar_masks = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other. broadcast to (B, S)
            _img_ar_masks += [False] * image_tokens.shape[1]
        img_ar_mask = jnp.array(_img_ar_masks)
        img_ar_mask = einops.repeat(img_ar_mask, "s -> b s", b=image_tokens.shape[0])

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            text_tokens = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(text_tokens)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs. reasoning tokens casual attention.
            text_ar_mask = obs.tokenized_reasoning_mask

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.concatenate([img_ar_mask, text_ar_mask], axis=1)
        return tokens, input_mask, ar_mask

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: CoTObservation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        # TODO: assume reasoning is already tokenized for compute_loss. Need to tokenize reasoning on-the-fly for inference.
        observation = preprocess_observation(preprocess_rng, observation, train=train)

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)

        if not self.lang_action_only:
            batch_shape = actions.shape[:-2]
            noise = jax.random.normal(noise_rng, actions.shape)
            time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
            time_expanded = time[..., None, None]
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            u_t = noise - actions
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(observation, x_t, time)
            suffix_ar_mask = einops.repeat(suffix_ar_mask, "s -> b s", b=suffix_tokens.shape[0])
            # one big forward pass of prefix + suffix at once. Reasoning is identical to prompt except for ar_mask.
            input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
            ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=1)
        else:
            suffix_tokens = None
            input_mask = prefix_mask
            ar_mask = prefix_ar_mask

        attn_mask = _pi0.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions
        )

        # TODO: should we jointly compute loss on the prompt?
        shift_labels = observation.tokenized_prompt[:, 1:]  # shape (B, max_len-1). TODO: contiguous?
        max_len = observation.tokenized_reasoning_mask.shape[1]
        shift_tokens = prefix_out[:, -max_len:-1, :]  # shape (B, max_len-1, D)
        shift_logits = self.PaliGemma.llm(shift_tokens, method="decode")  # shape (B, max_len-1, V)
        # compute cross-entropy loss on the reasoning tokens
        reasoning_and_pad_mask = jnp.logical_and(
            observation.tokenized_reasoning_mask[:, 1:],
            observation.tokenized_prompt_mask[:, 1:],
        )
        # If present, fold example_mask (B,) into the token mask (B, L-1)
        if getattr(observation, "example_mask", None) is not None:
            # Broadcast example_mask over the sequence length
            ex_mask = jnp.asarray(observation.example_mask)[..., None]
            token_mask = reasoning_and_pad_mask * ex_mask
        else:
            token_mask = reasoning_and_pad_mask

        # Optionally apply higher weights to numeric tokens within reasoning
        if getattr(observation, "tokenized_numeric_mask", None) is not None and self.number_token_weight != 1.0:
            numeric_shift = observation.tokenized_numeric_mask[:, 1:]
            weights = token_mask.astype(jnp.float32) * (
                1.0 + (self.number_token_weight - 1.0) * numeric_shift.astype(jnp.float32)
            )
            loss = cross_entropy_loss(
                shift_logits,
                shift_labels,
                mask=weights,
                axis=-1,
                train=True,
                per_example=True,
            )
        else:
            loss = cross_entropy_loss(
                shift_logits,
                shift_labels,
                mask=token_mask,
                axis=-1,
                train=True,
                per_example=True,
            )

        if not self.lang_action_only:
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            # Per-example action MSE reduced over (horizon, action_dim)
            loss = loss + jnp.mean(jnp.square(v_t - u_t), axis=(-1, -2))

        return loss

    # # Optional eval metrics for validation
    # def compute_eval_metrics(
    #     self, rng: at.KeyArrayLike, observation: Observation, actions: _model.Actions
    # ) -> dict[str, at.Array]:
    #     loss = self.compute_loss(rng, observation, actions, train=False)
    #     out = {"val_loss": jnp.mean(loss)}
    #     # Add a simple language perplexity estimate when reasoning tokens present
    #     if observation.tokenized_reasoning_mask is not None:
    #         try:
    #             prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    #             max_len = observation.tokenized_reasoning_mask.shape[1]
    #             positions = jnp.cumsum(prefix_mask, axis=1) - 1
    #             (prefix_out, _), _ = self.PaliGemma.llm([prefix_tokens, None], mask=make_attn_mask(prefix_mask, prefix_ar_mask), positions=positions)
    #             shift_tokens = prefix_out[:, -max_len:-1, :]
    #             logits = self.PaliGemma.llm(shift_tokens, method="decode")
    #             shift_labels = observation.tokenized_prompt[:, 1:]
    #             reasoning_and_pad_mask = jnp.logical_and(
    #                 observation.tokenized_reasoning_mask[:, 1:], observation.tokenized_prompt_mask[:, 1:]
    #             )
    #             nll = cross_entropy_loss(logits, shift_labels, mask=reasoning_and_pad_mask, axis=-1, train=False)
    #             out["val_lang_nll"] = nll
    #         except Exception:
    #             pass
    #     return out

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: CoTObservation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> _model.Actions:
        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, kv_cache = self._sample_reasoning_tokens(observation)

        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
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
        observation = preprocess_observation(None, observation, train=False)
        p_tokens, p_mask0, p_ar_mask0 = self.embed_prefix(observation)  # (B,Tp,D) + (B,Tp)
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

    def _sample_reasoning_tokens_2(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        max_decoding_steps: int | at.Int[at.Array, ""] = 256,
        temperature: float = 0.0,
    ) -> _model.Actions:
        # TODO: this is a hack to get the image keys.
        observation = _model.preprocess_observation(
            None, observation, train=False, image_keys=list(observation.images.keys())
        )

        # embed inputs
        prefix_token_embeddings, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)

        prefill_size = prefix_token_embeddings.shape[1]
        prefill_len = jnp.sum(prefix_mask, axis=-1)
        prefix_start = prefill_size - prefill_len

        # first fill KV cache with a forward pass of the prefix
        # pad attention mask to set the size of the KV cache (prefill_size + max_decoding_steps)
        prefix_attn_mask = jnp.pad(prefix_attn_mask, ((0, 0), (0, 0), (0, max_decoding_steps)))
        prefix_positions = jnp.cumsum(prefix_mask, axis=-1) - 1
        prefix_logits, kv_cache, _ = self.PaliGemma.llm(
            embedded_prefix=prefix_token_embeddings, mask=prefix_attn_mask, positions=prefix_positions, decode=True
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
            output_tokens = put_along_last_axis(output_tokens, jnp.broadcast_to(step, (token.shape[0], 1)), token)

            # Check for early stopping --> stop if all batch elements have EOS token
            has_eos = jnp.any(token == self.EOS_ID, axis=-1)
            all_eos = jnp.all(has_eos)

            # Decode one step
            token_embedding = self.PaliGemma.llm(token, embed_only=True)
            positions = prefill_len[:, None] + step + 1
            mask = jnp.logical_and(
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :] >= prefix_start[:, None, None],
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :]
                < (jnp.broadcast_to(prefill_size + step + 1, (prefix_start.shape[0], 1, 1))),
            )
            last_logit, kv_cache, _ = self.PaliGemma.llm(
                embedded_prefix=token_embedding, mask=mask, positions=positions, decode=True, kv_cache=cache
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

    def sample_reasoning(self, observation: CoTObservation):
        _, _, _, logits, t, _, _ = self._sample_reasoning_tokens(observation)
        return logits, t

        # output_tokens = self._sample_reasoning_tokens(observation)
        # return output_tokens
