# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""gemma reimplementation for big_vision.

We follow this einsum axis naming convention:
  B: batch
  T: query length
  S: k/v length
  N: num query heads
  K: num k/v heads
  G: num query heads per k/v head
  H: head dim
  D: d_model ("features")

Example Colab using the models via the PaliGemma decoding logic:
(internal link)

Doc locating the variable initializers in the original code and validating them:
(internal link)

This implementation does *not* currently support the local sliding attention
pattern used in the v2 models. But since we mostly use sequences <4096 tokens,
this shouldn't make any difference. Since RoPE embedding is used throughout,
it's unclear if there is any practical difference (other than wasting some
memory).
"""

from collections.abc import Sequence
import dataclasses
from typing import Literal

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
from openpi.models.gemma import PALIGEMMA_VOCAB_SIZE
from openpi.models.gemma import Block as _Block
from openpi.models.gemma import KVCache
from openpi.models.gemma import _apply_rope
import openpi.models.lora as lora
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding

from openpi_cot.models.gemma_common import Einsum as CommonEinsum
from openpi_cot.models.gemma_common import Embedder as CommonEmbedder
from openpi_cot.models.gemma_common import FeedForward as CommonFeedForward
from openpi_cot.models.gemma_common import RMSNorm as CommonRMSNorm
from openpi_cot.models.gemma_common import _gated_residual
from openpi_cot.models.gemma_common import _name


@dataclasses.dataclass
class Config:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    lora_configs: dict[str, lora.LoRAConfig] = dataclasses.field(default_factory=dict)
    query_pre_attn_norm: Literal["rsqrt_head_dim", "rsqrt_emb_per_head"] = "rsqrt_head_dim"
    final_logits_softcap: float | None = None  # new in gemma2
    attn_logits_softcap: float | None = None  # new in gemma2
    post_norms: bool = False  # new in gemma2
    param_dtype: str = "bfloat16"  # parameter storage dtype


Variant = Literal["gemma2_300m", "gemma2_2b", "gemma2_9b", "gemma2_27b", "gemma2_2b_lora"]


def get_config(variant: Variant) -> Config:
    """Returns config for specified gemma variant."""
    if variant == "gemma2_300m":
        # 311M params
        return Config(
            width=1024,
            depth=26,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=4,
            head_dim=256,
            final_logits_softcap=30.0,
            attn_logits_softcap=50.0,
            post_norms=False,
        )
    if variant == "gemma2_2b":
        return Config(
            width=2304,
            depth=26,
            mlp_dim=9216,
            num_heads=8,
            num_kv_heads=4,
            head_dim=256,
            final_logits_softcap=30.0,
            attn_logits_softcap=50.0,
            post_norms=True,
        )

    if variant == "gemma2_9b":
        return Config(
            width=3584,
            depth=42,
            mlp_dim=14_336,
            num_heads=16,
            num_kv_heads=8,
            head_dim=256,
            final_logits_softcap=30.0,
            attn_logits_softcap=50.0,
            post_norms=True,
        )
    if variant == "gemma2_27b":
        return Config(
            width=4608,
            depth=46,
            mlp_dim=36_864,
            num_heads=32,
            num_kv_heads=16,
            head_dim=128,
            query_pre_attn_norm="rsqrt_emb_per_head",
            final_logits_softcap=30.0,
            attn_logits_softcap=50.0,
            post_norms=True,
        )
    if variant == "gemma2_2b_lora":
        return Config(
            width=2304,
            depth=26,
            mlp_dim=9216,
            num_heads=8,
            num_kv_heads=4,
            head_dim=256,
            final_logits_softcap=30.0,
            attn_logits_softcap=50.0,
            post_norms=True,
            lora_configs={"attn": lora.LoRAConfig(rank=16, alpha=16.0), "ffn": lora.LoRAConfig(rank=16, alpha=16.0)},
        )

    raise ValueError(f"Unknown variant: {variant}")


# RMSNorm: Wrapper around common implementation with bfloat16 default for Gemma2
@at.typecheck
class RMSNorm(CommonRMSNorm):
    """RMSNorm with explicit bfloat16 parameter dtype (Gemma2 default).

    Inherits from common implementation and sets param_dtype="bfloat16" by default.
    """

    param_dtype: str = "bfloat16"


# Embedder: Wrapper around common implementation with bfloat16 default for Gemma2
@at.typecheck
class Embedder(CommonEmbedder):
    """Embedder module with explicit bfloat16 parameter dtype (Gemma2 default).

    Inherits from common implementation and sets param_dtype="bfloat16" by default.
    """

    param_dtype: str = "bfloat16"


# Einsum: Wrapper around common implementation with bfloat16 default for Gemma2
class Einsum(CommonEinsum):
    """Einsum with LoRA support and explicit bfloat16 parameter dtype (Gemma2 default).

    Inherits from common implementation and sets param_dtype="bfloat16" by default.
    """

    param_dtype: str = "bfloat16"
    init_fn: nn.initializers.Initializer = nn.initializers.zeros


# FeedForward: Wrapper around common implementation with bfloat16 default for Gemma2
class FeedForward(CommonFeedForward):
    """Feed forward module with explicit bfloat16 parameter dtype (Gemma2 default).

    Inherits from common implementation and sets param_dtype="bfloat16" by default.
    """

    param_dtype: str = "bfloat16"


@at.typecheck
class Attention(nn.Module):
    """Attention module."""

    configs: Sequence[Config]

    @nn.compact
    def __call__(self, xs, positions, attn_mask, kv_cache):
        # all experts must share the same head dim, num heads, and num kv heads for self-attention to work
        assert all(config.head_dim == self.configs[0].head_dim for config in self.configs)
        assert all(config.num_heads == self.configs[0].num_heads for config in self.configs)
        assert all(config.num_kv_heads == self.configs[0].num_kv_heads for config in self.configs)

        dtype = next(x.dtype for x in xs if x is not None)  # original dtype, could be half-precision

        qkvs = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                continue
            if config.num_kv_heads == config.num_heads:
                qkv_einsum = Einsum(
                    shape=(3, config.num_heads, config.width, config.head_dim),
                    name=_name("qkv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                    param_dtype=config.param_dtype,
                )
                qkvs.append(qkv_einsum("BSD,3KDH->3BSKH", x))
            else:
                q_einsum = Einsum(
                    shape=(config.num_heads, config.width, config.head_dim),
                    name=_name("q_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                    lora_config=config.lora_configs.get("attn"),
                    param_dtype=config.param_dtype,
                )
                q = q_einsum("BTD,NDH->BTNH", x)
                kv_einsum = Einsum(
                    shape=(2, config.num_kv_heads, config.width, config.head_dim),
                    name=_name("kv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                    param_dtype=config.param_dtype,
                )
                k, v = kv_einsum("BSD,2KDH->2BSKH", x)
                qkvs.append((q, k, v))

        q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))

        q = _apply_rope(q, positions=positions)

        if self.configs[0].query_pre_attn_norm == "rsqrt_head_dim":
            q *= self.configs[0].head_dim ** -0.5
        elif self.configs[0].query_pre_attn_norm == "rsqrt_emb_per_head":
            q *= (self.configs[0].width // self.configs[0].num_heads) ** -0.5
        else:
            raise ValueError(f"Unknown query_pre_attn_norm: {self.configs[0].query_pre_attn_norm}")

        k = _apply_rope(k, positions=positions)

        # should still be half-precision here (if input was half-precision)
        assert q.dtype == k.dtype == v.dtype == dtype

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            k = jnp.concatenate([cache_k, k], axis=1)
            v = jnp.concatenate([cache_v, v], axis=1)

        q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=self.configs[0].num_kv_heads)
        logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)

        if self.configs[0].attn_logits_softcap:
            logits = jnp.tanh(logits / self.configs[0].attn_logits_softcap)
            logits = logits * self.configs[0].attn_logits_softcap

        if attn_mask.shape != (q.shape[0], 1, q.shape[1], k.shape[1]):
            raise ValueError(
                f"Attention mask with shape {attn_mask.shape} but shapes for q and k are: {q.shape} and {k.shape}"
            )

        # big_neg = jnp.finfo(logits.dtype).min
        big_neg = -2.3819763e38  # See gemma/modules.py
        masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)

        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)

        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")

        out = []
        start = 0
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                end = start + x.shape[1]
                out_einsum = Einsum(
                    shape=(config.num_heads, config.head_dim, config.width),
                    name=_name("attn_vec_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                    lora_config=config.lora_configs.get("attn"),
                    param_dtype=config.param_dtype,
                )
                out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))
                start = end
            else:
                out.append(None)

        return out, (k, v)


@at.typecheck
class Block(_Block):
    """Transformer block."""

    configs: tuple[Config, ...]  # Override parent class annotation with gemma2.Config
    post_norms: bool = True  # new in gemma2

    @nn.compact
    def __call__(self, xs, kv_cache, positions, attn_mask, adarms_cond, deterministic=True):  # noqa: FBT002
        xs = sharding.activation_sharding_constraint(xs)
        drop = nn.Dropout(self.dropout, self.dropout_bdims) if self.dropout else lambda x, _: x

        attn = Attention(configs=self.configs, name="attn")

        pre_attn = []
        gates = []
        for i, x in enumerate(xs):
            if x is not None:
                x, gate = RMSNorm(name=_name("pre_attention_norm", i), param_dtype=self.configs[i].param_dtype)(
                    x, adarms_cond[i]
                )
            pre_attn.append(x)
            gates.append(gate if x is not None else None)

        pre_attn = sharding.activation_sharding_constraint(pre_attn)
        post_attn, kv_cache = attn(pre_attn, positions, attn_mask, kv_cache)
        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)

        # Apply post_attention_norm only to the first expert if post_norms is enabled
        if self.post_norms:
            post_attn_normed = []
            for i, x in enumerate(post_attn):
                if x is not None and i == 0:
                    x, _ = RMSNorm(name="post_attention_norm", param_dtype=self.configs[i].param_dtype)(x, None)  # noqa: PLW2901
                post_attn_normed.append(x)
            post_attn = post_attn_normed

        post_attn = sharding.activation_sharding_constraint(post_attn)
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, post_attn, gates, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)

        out = []
        gates = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                x, gate = RMSNorm(name=_name("pre_ffw_norm", i), param_dtype=config.param_dtype)(x, adarms_cond[i])  # noqa: PLW2901
                x = FeedForward(  # noqa: PLW2901
                    features=config.width,
                    hidden_dim=config.mlp_dim,
                    name=_name("mlp", i),
                    lora_config=config.lora_configs.get("ffn"),
                    param_dtype=config.param_dtype,
                )(x)
            out.append(x)
            gates.append(gate if x is not None else None)

        out = sharding.activation_sharding_constraint(out)
        out = jax.tree.map(lambda x: drop(x, deterministic), out)

        # Apply post_ffw_norm only to the first expert if post_norms is enabled
        if self.post_norms:
            out_normed = []
            for i, x in enumerate(out):
                if x is not None and i == 0:
                    x, _ = RMSNorm(name="post_ffw_norm", param_dtype=self.configs[i].param_dtype)(x, None)  # noqa: PLW2901
                out_normed.append(x)
            out = out_normed

        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, out, gates, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)

        return xs, kv_cache


@at.typecheck
class Module(nn.Module):
    """Transformer model, supporting a mixture of different weights for different tokens."""

    configs: Sequence[Config]  # list of configs, one for each expert
    embed_dtype: str

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()  # Every float is dropped independently.
    adarms: bool = False

    def setup(self):
        # all experts must have the same depth
        assert all(config.depth == self.configs[0].depth for config in self.configs)

        self.embedder = Embedder(
            vocab_size=PALIGEMMA_VOCAB_SIZE,
            embed_dim=self.configs[0].width,  # embedder for first expert only
            param_dtype=self.configs[0].param_dtype,
            name="embedder",
        )
        block_cls = nn.remat(
            Block,
            prevent_cse=False,
            static_argnums=(5,),  # 0=self, 6=deterministic
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(
                0,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
            ),  # 0=kv_cache, 1=positions, 2=mask, 3=adarms_cond, 4=deterministic
            length=self.configs[0].depth,
        )(
            configs=self.configs,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
            post_norms=self.configs[0].post_norms,
        )
        self.final_norms = [
            RMSNorm(name=_name("final_norm", i), param_dtype=self.configs[i].param_dtype)
            for i in range(len(self.configs))
        ]

    @at.typecheck
    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        return self.embedder.encode(tokens).astype(self.embed_dtype)

    @at.typecheck
    def __call__(
        self,
        # list of token arrays, one for each expert, or None if that expert should not be run
        embedded: Sequence[at.Float[at.Array, "b _t _d"] | None],
        positions: at.Int[at.Array, "b t"],
        mask: at.Bool[at.Array, "b t s"],
        adarms_cond: Sequence[at.Float[at.Array, "b _d"] | None] | None = None,
        *,
        kv_cache: KVCache | None = None,
        deterministic: bool = True,
    ) -> tuple[Sequence[at.Float[at.Array, "b _t _d"] | None], KVCache]:
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        mask = jnp.asarray(mask)[:, None, :, :]
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)

        embedded, kv_cache = self.layers(embedded, kv_cache, positions, mask, adarms_cond, deterministic)

        assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)

        return [
            f(e, a)[0] if e is not None else e for f, e, a in zip(self.final_norms, embedded, adarms_cond, strict=True)
        ], kv_cache

    def init(self, use_adarms: Sequence[bool]):
        """Convenience method for initializing all parameters, necessary due to the quirks of linen."""
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))
        self(
            [jnp.zeros((1, 1, c.width)) for c in self.configs],
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),
            adarms_cond=[jnp.zeros((1, c.width)) if u else None for u, c in zip(use_adarms, self.configs, strict=True)],
        )
