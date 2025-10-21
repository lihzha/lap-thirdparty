import jax
import jax.numpy as jnp
import flax.nnx as nnx
import openpi_cot.models.pi_cot_config as _pi_cot_config

# Assume your provided code is saved in a file named `pi_cot_model.py`
from openpi_cot.models.pi_cot import PiCoT

# 1. Define Configuration Parameters for the PiCoT model
# We set the variants as requested and provide reasonable default values for other parameters.
# Literal["gemma2_300m", "gemma2_2b"]
#'paligemma_variant': 'gemma3_4b',
    #'action_expert_variant': 'gemma3_300m'
config_params = {
    'paligemma_variant': 'gemma3_4b',
    'action_expert_variant': 'gemma3_300m',
    'dtype': 'bfloat16',          # The config expects a string for the dtype
    'action_dim': 7,              # Example: 6-DoF arm + 1-DoF gripper
    'action_horizon': 8,          # Example: Predict 8 future action steps
    'max_token_len': 256,         # Example: Maximum length for text sequences
    'aug_wrist_image': True,      # Enable data augmentation for the wrist camera
    'pi05': True,                 # Use the Pi0.5 architecture variant (AdaRMS)
    # By default, use_bimanual is False, so it will use 2 cameras.
    # Training flags can be set here to override defaults from the config class
    'enable_action_training': True,
    'enable_langact_training': True,
    'enable_prediction_training': False,
}

# 2. Create the PiCoTConfig instance
# We use ** to unpack the dictionary into keyword arguments for the constructor.
# Note: This assumes PiCoTConfig accepts these as keyword arguments.
print("Creating PiCoTConfig...")
picot_config = _pi_cot_config.PiCoTConfig(**config_params)
print("Config created successfully.")

# 3. Create NNX Rngs for initialization
# This provides the necessary random keys for initializing model parameters (e.g., Linear layers).
print("Creating JAX PRNGKey for initialization...")
rng_key = jax.random.key(0)  # Use seed 0 for reproducibility


# 4. Instantiate the PiCoT model
# The model's parameters will be lazily initialized inside the constructor
# due to the use of nnx_bridge.ToNNX with lazy_init.
print("\nInstantiating the PiCoT model...")
try:
    model: PiCoT = picot_config.create(rng=rng_key)
    #model = PiCoT(config=picot_config, rngs=rng_key)
    print("✅ Model instantiated successfully!")

    # You can now inspect the model structure.
    # The summary will show the initialized components.
    print("\nModel structure summary:")
    print(model)

except Exception as e:
    print(f"❌ An error occurred during model instantiation: {e}")
   