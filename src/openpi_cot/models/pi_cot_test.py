import jax
import logging

# Set up basic logging to see potential warnings or errors from the libraries
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("openpi")
logger.setLevel(logging.INFO)


# --- Assume your provided code files are saved as follows ---
# 1. The main model code is in `pi_cot_model.py`
# 2. The config code is in `pi_cot_config.py`

# We import the necessary classes from your files
from openpi_cot.models.pi_cot_config import PiCoTConfig
from openpi_cot.models.pi_cot import PiCoT


# 1. Define Configuration Parameters
# We specify the exact variants you requested and provide sensible defaults
# for other required parameters.
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
# The dataclass constructor accepts our dictionary of parameters.
print("Creating PiCoTConfig...")
picot_config = PiCoTConfig(**config_params)
print(f"Config created successfully with paligemma_variant='{picot_config.paligemma_variant}' "
      f"and action_expert_variant='{picot_config.action_expert_variant}'.")

# 3. Create a JAX PRNGKey for initialization
# The .create() method expects a raw JAX key.
print("Creating JAX PRNGKey for initialization...")
rng_key = jax.random.key(0)  # Use seed 0 for reproducibility

# 4. Instantiate the PiCoT model using the config's create() method
# This is the recommended way to build the model. It handles passing the
# config and creating the Rngs object internally.
print("\nInstantiating the PiCoT model using config.create()...")
try:
    # This single call will initialize the entire model structure.
    # It will call the get_gemma3_config function inside your PiCoT class
    # to fetch the configurations for 'gemma_3_4b' and 'gemma_3_300m'.
    model: PiCoT = picot_config.create()
    print("✅ Model instantiated successfully!")

    # You can now inspect the model's structure.
    # The summary will show the initialized components and their parameters.
    print("\nModel structure summary:")
    # The summary() method is available on nnx.Module objects
    print(model.summary())

except Exception as e:
    print(f"❌ An error occurred during model instantiation: {e}")
    print("\n--- Troubleshooting ---")
    print("1. Ensure that the `openpi_cot` and `openpi` packages are correctly installed in your environment.")
    print("2. Check that the function `openpi_cot.models.gemma3.get_config` supports the variants 'gemma_3_4b' and 'gemma_3_300m'.")
    print("3. Verify that all dependencies like JAX, Flax, and Einops are installed.")