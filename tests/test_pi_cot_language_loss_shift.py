import jax
import jax.numpy as jnp
from openpi.models import model as _model

import openpi_cot.models.pi_cot as pi_cot
from openpi_cot.models.model_adapter import CoTObservation
from openpi_cot.models.pi_cot_config import PiCoTConfig


class _DummyLLM:
    def __init__(self, logits: jnp.ndarray, expected_seq_len: int):
        self._logits = logits
        self._expected_seq_len = expected_seq_len

    def __call__(self, pre_logits, method=None):
        assert method == "decode"
        assert pre_logits.shape[1] == self._expected_seq_len
        return self._logits


def test_compute_language_loss_shift_one_right_padded():
    vocab_size = 8
    pi_cot.PALIGEMMA_VOCAB_SIZE = vocab_size

    config = PiCoTConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        max_token_len=6,
        enable_action_training=False,
        enable_langact_training=True,
        enable_prediction_training=False,
        enable_vqa_training=False,
        pi05=True,
        discrete_state_input=True,
    )
    model = config.create(jax.random.PRNGKey(0))

    batch = 1
    height, width = _model.IMAGE_RESOLUTION
    image = jnp.zeros((batch, height, width, 3), dtype=jnp.float32)
    images = dict.fromkeys(config.image_keys, image)
    image_masks = {key: jnp.ones((batch,), dtype=bool) for key in config.image_keys}
    state = jnp.zeros((batch, config.action_dim), dtype=jnp.float32)

    tokenized_prompt = jnp.array([[1, 2, 3, 4, 0, 0]], dtype=jnp.int32)
    tokenized_prompt_mask = jnp.array([[1, 1, 1, 1, 0, 0]], dtype=bool)
    tokenized_langact_mask = jnp.array([[1, 1, 1, 1, 0, 0]], dtype=bool)
    token_loss_mask = jnp.array([[1, 1, 1, 1, 0, 0]], dtype=bool)

    observation = CoTObservation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
        token_loss_mask=token_loss_mask,
        tokenized_langact_mask=tokenized_langact_mask,
        sample_mask=None,
    )

    seq_len = tokenized_prompt.shape[1] - 1
    logits = jnp.full((batch, seq_len, vocab_size), -10.0)
    target_tokens = tokenized_prompt[0, 1:]
    for idx in range(seq_len):
        if idx < 3:
            logits = logits.at[0, idx, target_tokens[idx]].set(10.0)
        else:
            logits = logits.at[0, idx, 1].set(10.0)

    model.PaliGemma.llm = _DummyLLM(logits, expected_seq_len=seq_len)

    prefix_pre_logits = jnp.zeros((batch, seq_len + 1, 4), dtype=jnp.float32)
    per_sample_loss, _ = model._compute_language_loss(observation, prefix_pre_logits)

    targets = jax.nn.one_hot(tokenized_prompt[:, 1:], vocab_size)
    token_mask = jnp.logical_and(
        tokenized_langact_mask[:, 1:],
        jnp.logical_and(tokenized_prompt_mask[:, 1:], token_loss_mask[:, 1:]),
    )
    logp = jax.nn.log_softmax(logits, axis=-1)
    token_pplx = jnp.sum(targets * logp, axis=-1)
    expected = -jnp.sum(token_pplx * token_mask, axis=-1) / jnp.clip(jnp.sum(token_mask, -1), 1)

    assert int(jnp.sum(token_mask)) == 3
    assert jnp.allclose(per_sample_loss, expected)


if __name__ == "__main__":
    test_compute_language_loss_shift_one_right_padded()
