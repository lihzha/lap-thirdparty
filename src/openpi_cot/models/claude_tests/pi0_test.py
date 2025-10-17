import flax.nnx as nnx
import jax

import openpi_cot.models.claude_tests.pi0_config_gemma3 as _pi0_config


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)

# --- NEW main function to inspect the model ---

def main():
    """
    Initializes the model for full fine-tuning and prints all its parameters,
    their shapes, and their sizes.
    """
    print("--- Loading Full Fine-Tuning Model Configuration ---")
    config = _pi0_config.Pi0Config()

    print("--- Initializing Model and Parameters ---")
    # To get parameter values, we must initialize the model, not just evaluate its shape.
    # We provide a random key for deterministic initialization.
    key = jax.random.key(0)
    # The `create` method needs an Rngs object to initialize parameters.
    model = config.create(rng=key)



    # Extract all parameters (not just frozen ones) from the model.
    # We use nnx.Param to filter for only trainable parameters.
    params = nnx.state(model, nnx.Param).flat_state()

    print(f"\nFound {len(params)} parameter arrays in the model.")
    print("-" * 80)

    total_params = 0
    # Iterate through the flat dictionary of parameters
    for name, value in params.items():
        # value is a JAX array, so .size gives the total number of elements
        param_count = value.shape
        total_params += param_count
        # Print formatted info: name, shape, and element count
        print(f"- {name:<80} | Shape: {str(value.shape):<20} | Size: {param_count:,}")

    print("=" * 80)
    print(f"Total Trainable Parameters: {total_params:,}")


if __name__ == "__main__":
    main()