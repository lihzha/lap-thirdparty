import jax
import jax.numpy as jnp
import re
import flax.traverse_util
from collections import defaultdict

# --- Step 1: Import the necessary modules ---

# Import the weight loader from your other file
from openpi_cot.training.weight_loaders import WeightLoader, restore_params, Gemma3WeightLoader, _merge_params, Gemma3ScanCompatibleWeightLoader

import openpi_cot.models.gemma3 as gemma3
import openpi.models.model as _model


# --- Step 2: Define model config (same as before) ---
VARIANT = "gemma3_4b"
action_expert = "gemma3_300m"
print(f"Setting up model for variant: {VARIANT} and action expert: {action_expert}")
config_gemma3 = gemma3.get_config(VARIANT)
#config_action_expert = gemma3.get_config(action_expert)
configs = [config_gemma3]


#--- Step 3: Instantiate the UNMODIFIED Model ---
model = gemma3.Module(
    configs=configs,
    embed_dtype="bfloat16"
)

# --- Step 4: Create Dummy Data and Initialize (same as before) ---
key = jax.random.PRNGKey(0)
batch_size, seq_len = 1, 128
dummy_embedded = [jnp.zeros((batch_size, seq_len, configs[0].width), dtype=jnp.bfloat16)]
dummy_positions = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
dummy_mask = jnp.ones((batch_size, seq_len, seq_len), dtype=jnp.bool_)
dummy_adarms_cond = [None] * len(configs)

print("Initializing model to create the parameter structure...")

initialized_model_params = model.init(
    key,
    embedded=dummy_embedded,
    positions=dummy_positions,
    mask=dummy_mask,
    adarms_cond=dummy_adarms_cond
)['params']
print("Model parameter structure created successfully.")


print("Starting weight loading with key remapping...")
# The loader will now handle the translation automatically.

# --- Step 5: Use the NEW Key-Matching Loader ---
gemma3_weights_path = "src/openpi_cot/ckpts/gemma3-4b-it/"
# Instantiate our custom loader directly.
loader = Gemma3ScanCompatibleWeightLoader(params_path=gemma3_weights_path)

loaded_params = loader.load(params=initialized_model_params)
print("✅ Gemma 3 weights loaded successfully using the key-matching adapter!")