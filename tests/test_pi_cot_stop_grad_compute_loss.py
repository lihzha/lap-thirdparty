import dataclasses

import jax
import jax.numpy as jnp
from openpi.models import model as _model

from openpi_cot.models.model_adapter import CoTObservation
from openpi_cot.models.model_adapter import preprocess_observation
from openpi_cot.models.pi_cot_config import PiCoTConfig


def compute_loss(
    rng,
    observation,
    model,
    actions: _model.Actions,
    train: bool = True,
):
    preprocess_rng, _, noise_rng, time_rng = jax.random.split(rng, 4)

    # Preprocess observation (will skip augmentation for VQA samples if vqa_mask is provided)
    observation = preprocess_observation(
        preprocess_rng,
        observation,
        train=train,
        image_keys=model.image_keys,
        aug_wrist_image=model.aug_wrist_image,
        aggresive_aug=model.aggresive_aug,
        vqa_mask=None,
    )

    # Build prefix for langact/action losses (first frame + text)
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)

    suffix_inputs = model.prepare_suffix(observation, actions, noise_rng, time_rng)
    prefix_mask_action = model._build_prefix_action_mask(prefix_mask, observation)
    combined_mask = model._build_combined_attention_mask(
        prefix_mask,
        prefix_ar_mask,
        prefix_mask_action,
        suffix_inputs["suffix_mask"],
        suffix_inputs["suffix_ar_mask"],
    )
    combined_positions = model._build_combined_positions(prefix_mask, prefix_mask_action, suffix_inputs["suffix_mask"])

    pre_logits, _ = model.llm(
        embedded=[prefix_tokens, suffix_inputs["suffix_tokens"]],
        positions=combined_positions,
        mask=combined_mask,
        adarms_cond=[None, suffix_inputs["adarms_cond"]],
    )

    metrics = {}
    total_per_sample_loss = 0

    combined_langact_mask = observation.sample_mask
    lang_loss, lang_metrics = model._compute_language_loss(
        observation,
        pre_logits[0],
        sample_mask=combined_langact_mask,
        verbose_mode=False,
    )
    metrics.update(lang_metrics)

    # metrics.update(
    #     _compute_sample_specific_metrics(
    #         per_sample_loss=lang_loss,
    #         lang_metrics=lang_metrics,
    #         sample_mask=combined_langact_mask,
    #         prefix="langact_",
    #         verbose_mode=False,
    #     )
    # )
    total_per_sample_loss += lang_loss

    suffix_out = pre_logits[1]
    action_loss, action_metrics = model._compute_action_loss(suffix_out, suffix_inputs["u_t"])
    total_per_sample_loss += action_loss
    metrics.update(action_metrics)
    # final_loss = jnp.mean(total_per_sample_loss)

    return jnp.mean(action_loss), jnp.mean(lang_loss)


def _build_observation(config: PiCoTConfig) -> CoTObservation:
    batch = 4
    height, width = _model.IMAGE_RESOLUTION
    image = jnp.linspace(-1.0, 1.0, num=batch * height * width * 3, dtype=jnp.float32).reshape(
        (batch, height, width, 3)
    )
    images = dict.fromkeys(config.image_keys, image)
    image_masks = {key: jnp.ones((batch,), dtype=bool) for key in config.image_keys}
    state = jnp.zeros((batch, config.action_dim), dtype=jnp.float32)

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


def _compute_grads(model, rng, observation, actions):
    action_loss, langact_loss = compute_loss(rng, observation, model, actions, train=True)

    grad_action_prefix = jax.grad(_action_loss, argnums=0)(prefix_tokens, suffix_tokens)
    grad_action_suffix = jax.grad(_action_loss, argnums=1)(prefix_tokens, suffix_tokens)
    grad_lang_prefix = jax.grad(_lang_loss, argnums=0)(prefix_tokens, suffix_tokens)
    grad_lang_suffix = jax.grad(_lang_loss, argnums=1)(prefix_tokens, suffix_tokens)

    return {
        "action_wrt_prefix": grad_action_prefix,
        "action_wrt_suffix": grad_action_suffix,
        "lang_wrt_prefix": grad_lang_prefix,
        "lang_wrt_suffix": grad_lang_suffix,
    }


def _make_config(stop_action_to_vlm_grad: bool) -> PiCoTConfig:
    return PiCoTConfig(
        paligemma_variant="gemma_300m",
        action_expert_variant="gemma_300m",
        max_token_len=100,
        enable_action_training=True,
        enable_langact_training=True,
        enable_prediction_training=False,
        enable_vqa_training=False,
        stop_action_to_vlm_grad=stop_action_to_vlm_grad,
        pi05=True,
        discrete_state_input=True,
    )


def test_stop_action_to_vlm_grad_with_compute_loss_inputs():
    config_stop = _make_config(stop_action_to_vlm_grad=True)
    config_allow = dataclasses.replace(config_stop, stop_action_to_vlm_grad=False)

    observation = _build_observation(config_stop)
    actions = _build_actions(config_stop)

    model_stop = config_stop.create(jax.random.PRNGKey(0))
    model_allow = config_allow.create(jax.random.PRNGKey(0))

    grads_stop = _compute_grads(model_stop, jax.random.PRNGKey(1), observation, actions)
    grads_allow = _compute_grads(model_allow, jax.random.PRNGKey(1), observation, actions)

    print("Grads stop: ", grads_stop)
    print("Grads allow: ", grads_allow)

    assert bool(jnp.all(grads_stop["action_wrt_prefix"] == 0))
    assert float(jnp.linalg.norm(grads_allow["action_wrt_prefix"])) > 1e-6

    assert bool(jnp.all(grads_stop["lang_wrt_suffix"] == 0))
    assert bool(jnp.all(grads_allow["lang_wrt_suffix"] == 0))

    breakpoint()

    assert float(jnp.linalg.norm(grads_stop["lang_wrt_prefix"])) > 1e-6
    assert float(jnp.linalg.norm(grads_stop["action_wrt_suffix"])) > 1e-6


if __name__ == "__main__":
    test_stop_action_to_vlm_grad_with_compute_loss_inputs()
