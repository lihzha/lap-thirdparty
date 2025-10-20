import flax.nnx as nnx
import jax
import functools

import openpi_cot.models.pi0_config_gemma3 as _pi0_config


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
    config = _pi0_config.Pi0Config(
        action_dim=7,
            action_horizon=10,
            max_token_len=32,
            pi05=False
    )

    print("--- Initializing Model and Parameters ---")
    # To get parameter values, we must initialize the model, not just evaluate its shape.
    # We provide a random key for deterministic initialization.
    key = jax.random.key(0)
    # The `create` method needs an Rngs object to initialize parameters.
    model = config.create(rng=key)



    # NEW, ROBUST HELPER FUNCTION
    def get_nested_module(root_module, path):
        """
        Accesses a nested module using a path tuple, handling both
        attribute and dictionary/list item access.
        """
        current_node = root_module
        for name in path:
            if isinstance(current_node, (dict, list, tuple)):
                # If the current node is a collection, use item access.
                # The key might be a string representing an integer index.
                try:
                    key = int(name) if name.isdigit() else name
                    current_node = current_node[key]
                except (KeyError, IndexError):
                    # Fallback for string keys if int conversion fails/is out of bounds
                    current_node = current_node[name]
            else:
                # Otherwise, it's a module; use attribute access.
                current_node = getattr(current_node, name)
        return current_node


    # 1. Get the flattened state.
    params = nnx.state(model, nnx.Param).flat_state()

    print(f"\nFound {len(params)} trainable parameter arrays.")
    print("\nDetailed Parameter Information:")
    print("-" * 120)
    print(f"{'Full Parameter Path':<65} | {'Owner Class':<20} | {'Shape':<20} | {'Size'}")
    print("-" * 120)

    total_params = 0
    # 2. Iterate through the flat dictionary of parameters.
    for path, var_state in params.items():
        # The owner's path is the full path minus the last element (the param name)
        owner_path = path[:-1]

        # Use our NEW, robust helper to get the actual module object
        if owner_path: # Ensure path is not empty
             owner_module = get_nested_module(model, owner_path)
        else: # The parameter is directly on the top-level model
            owner_module = model
            
        owner_class = owner_module.__class__.__name__

        # Extract parameter info
        jax_array = var_state.value
        shape = jax_array.shape
        size = jax_array.size
        total_params += size

        full_path_str = '.'.join(path)

        # Print the detailed, formatted line
        print(f"{full_path_str:<65} | {owner_class:<20} | {str(shape):<20} | {size:,}")

    print("=" * 120)
    print(f"Total Trainable Parameters: {total_params:,}")


if __name__ == "__main__":
    main()