import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from openpi.models import model as _model
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

from openpi_cot.models.model_adapter import CoTObservation
from openpi_cot.models.pi_cot_config import PiCoTConfig

_VLM_FILTER = nnx.Any(
    nnx.All(
        nnx_utils.PathRegex(".*llm.*"),
        nnx.Not(nnx_utils.PathRegex(".*_1.*")),
    ),
    nnx_utils.PathRegex(".*img.*"),
)
_ACTION_EXPERT_FILTER = nnx_utils.PathRegex(".*llm.*_1.*")


def _build_observation(config: PiCoTConfig) -> CoTObservation:
    batch = 4
    height, width = _model.IMAGE_RESOLUTION
    image = jnp.linspace(-1.0, 1.0, num=batch * height * width * 3, dtype=jnp.float32).reshape(
        (batch, height, width, 3)
    )
    images = dict.fromkeys(config.image_keys, image)
    image_masks = {key: jnp.ones((batch,), dtype=bool) for key in config.image_keys}
    state = jnp.ones((batch, config.action_dim), dtype=jnp.float32)

    prompt_len = config.max_token_len
    pad_len = 10
    promptonly_len = 20

    tokenized_prompt = jnp.arange(prompt_len, dtype=jnp.int32)[None, :] + 2
    tokenized_prompt = jnp.tile(tokenized_prompt, (batch, 1))
    prompt_mask = jnp.ones((batch, prompt_len), dtype=bool)
    tokenized_langact_mask = jnp.ones((batch, prompt_len), dtype=bool)
    token_loss_mask = jnp.ones((batch, prompt_len), dtype=bool)

    prompt_mask = prompt_mask.at[:, -pad_len:].set(False)
    tokenized_langact_mask = tokenized_langact_mask.at[:, :promptonly_len].set(False)
    tokenized_langact_mask = tokenized_langact_mask.at[:, -pad_len:].set(False)
    token_loss_mask = token_loss_mask.at[:, :promptonly_len].set(False)
    token_loss_mask = token_loss_mask.at[:, -pad_len:].set(False)

    sample_mask = jnp.arange(batch) % 2 == 0

    return CoTObservation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=prompt_mask,
        token_loss_mask=token_loss_mask,
        tokenized_langact_mask=tokenized_langact_mask,
        sample_mask=sample_mask,
    )


def _build_actions(config: PiCoTConfig) -> _model.Actions:
    total = config.action_horizon * config.action_dim * 4
    actions = jnp.linspace(-0.5, 0.5, num=total, dtype=jnp.float32).reshape(
        (4, config.action_horizon, config.action_dim)
    )
    return actions


def _make_config(
    *,
    stop_action_to_vlm_grad: bool,
    enable_langact_training: bool,
    action_loss_weight: float,
) -> PiCoTConfig:
    return PiCoTConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        max_token_len=100,
        enable_action_training=True,
        enable_langact_training=enable_langact_training,
        enable_prediction_training=False,
        enable_vqa_training=False,
        action_loss_weight=action_loss_weight,
        stop_action_to_vlm_grad=stop_action_to_vlm_grad,
        pi05=True,
        discrete_state_input=True,
    )


def _grad_norm(grads: nnx.State, filter: nnx.filterlib.Filter) -> jnp.ndarray:
    filtered = grads.filter(filter).flat_state()
    if not filtered:
        raise AssertionError("No gradient leaves matched the filter.")
    total = 0.0
    for value in filtered.values():
        leaf = value.value if hasattr(value, "value") else value
        total += jnp.sum(jnp.square(leaf))
    return jnp.sqrt(total)


def _compute_grads(model, rng, observation, actions) -> nnx.State:
    stage_config = None
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        observation: CoTObservation | _model.Observation,
        actions: _model.Actions,
    ):
        loss, metrics = model.compute_loss(rng, observation, actions, train=True, stage_config=stage_config)
        return loss, metrics

    train_rng = jax.random.fold_in(rng, 0)
    diff_state = nnx.DiffState(0, nnx.Param)
    (_, _), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )
    return grads


def test_compute_loss_blocks_action_loss_to_vlm_params():
    config_stop = _make_config(
        stop_action_to_vlm_grad=True,
        enable_langact_training=False,
        action_loss_weight=1.0,
    )
    config_allow = dataclasses.replace(config_stop, stop_action_to_vlm_grad=False)

    observation = _build_observation(config_stop)
    actions = _build_actions(config_stop)

    model_stop = config_stop.create(jax.random.PRNGKey(0))
    model_allow = config_allow.create(jax.random.PRNGKey(0))

    grads_stop = _compute_grads(model_stop, jax.random.PRNGKey(1), observation, actions)
    grads_allow = _compute_grads(model_allow, jax.random.PRNGKey(1), observation, actions)

    # inside test_compute_loss_blocks_action_loss_to_vlm_params
    prefix_tokens, prefix_mask, prefix_ar_mask = model_allow.embed_prefix(observation)
    suffix_inputs = model_allow.prepare_suffix(observation, actions, jax.random.PRNGKey(0), jax.random.PRNGKey(1))
    combined_mask = model_allow._build_combined_attention_mask(
        prefix_mask,
        prefix_ar_mask,
        model_allow._build_prefix_action_mask(prefix_mask, observation),
        suffix_inputs["suffix_mask"],
        suffix_inputs["suffix_ar_mask"],
    )
    combined_positions = model_allow._build_combined_positions(
        prefix_mask, model_allow._build_prefix_action_mask(prefix_mask, observation), suffix_inputs["suffix_mask"]
    )

    def _action_loss_from_prefix(prefix_tokens):
        pre_logits, _ = model_allow.PaliGemma.llm(
            embedded=[prefix_tokens, suffix_inputs["suffix_tokens"]],
            positions=combined_positions,
            mask=combined_mask,
            adarms_cond=[None, suffix_inputs["adarms_cond"]],
        )
        suffix_out = pre_logits[1]
        loss, _ = model_allow._compute_action_loss(suffix_out, suffix_inputs["u_t"])
        return jnp.mean(loss)

    print("grad wrt prefix_tokens:", jnp.linalg.norm(jax.grad(_action_loss_from_prefix)(prefix_tokens)))

    nonzero = {
        k: float(jnp.linalg.norm(v.value if hasattr(v, "value") else v))
        for k, v in grads_allow.flat_state().items()
        if float(jnp.linalg.norm(v.value if hasattr(v, "value") else v)) > 0
    }
    print([k for k in nonzero if "llm" in k][:20])

    print(f"Grad norm VLM (stop): {float(_grad_norm(grads_stop, _VLM_FILTER))}")
    print(f"Grad norm VLM (allow): {float(_grad_norm(grads_allow, _VLM_FILTER))}")
    print(f"Grad norm Action Expert (stop): {float(_grad_norm(grads_stop, _ACTION_EXPERT_FILTER))}")

    assert float(_grad_norm(grads_stop, _VLM_FILTER)) == 0
    assert float(_grad_norm(grads_allow, _VLM_FILTER)) > 1e-5
    assert float(_grad_norm(grads_stop, _ACTION_EXPERT_FILTER)) > 1e-5


def test_compute_loss_lang_loss_does_not_reach_action_expert():
    config = _make_config(
        stop_action_to_vlm_grad=False,
        enable_langact_training=True,
        action_loss_weight=0.0,
    )

    observation = _build_observation(config)
    actions = _build_actions(config)

    model = config.create(jax.random.PRNGKey(0))
    grads = _compute_grads(model, jax.random.PRNGKey(1), observation, actions)

    print(f"Grad norm VLM: {float(_grad_norm(grads, _VLM_FILTER))}")
    print(f"Grad norm Action Expert: {float(_grad_norm(grads, _ACTION_EXPERT_FILTER))}")

    assert float(_grad_norm(grads, _ACTION_EXPERT_FILTER)) == 0
    assert float(_grad_norm(grads, _VLM_FILTER)) > 1e-5


def test_compute_loss_self_attention_untouched_with_stop_grad():
    config = _make_config(
        stop_action_to_vlm_grad=True,
        enable_langact_training=True,
        action_loss_weight=1.0,
    )

    observation = _build_observation(config)
    actions = _build_actions(config)

    model = config.create(jax.random.PRNGKey(0))
    grads = _compute_grads(model, jax.random.PRNGKey(1), observation, actions)

    print(f"Grad norm VLM: {float(_grad_norm(grads, _VLM_FILTER))}")
    print(f"Grad norm Action Expert: {float(_grad_norm(grads, _ACTION_EXPERT_FILTER))}")

    assert float(_grad_norm(grads, _VLM_FILTER)) > 1e-5
    assert float(_grad_norm(grads, _ACTION_EXPERT_FILTER)) > 1e-5


if __name__ == "__main__":
    test_compute_loss_blocks_action_loss_to_vlm_params()
    test_compute_loss_lang_loss_does_not_reach_action_expert()
    test_compute_loss_self_attention_untouched_with_stop_grad()
