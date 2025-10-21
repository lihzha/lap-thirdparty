import jax
import jax.numpy as jnp
import re
import flax.traverse_util
from collections import defaultdict

# --- Step 1: Import the necessary modules ---

# Import the weight loader from your other file
from openpi_cot.training.weight_loaders import WeightLoader, restore_params, Gemma3WeightLoader, _merge_params
import openpi_cot.models.gemma3 as gemma3
import openpi.models.model as _model

class Gemma3ScanCompatibleLoader(WeightLoader):
    param_path: str

    def __init__(self, params_path: str):
        self.param_path = params_path

    def load(self, params: dict) -> dict:
        print("Loading Gemma 3 weights using Gemma3ScanCompatibleLoader...")
        loaded_params = Gemma3WeightLoader(self.param_path).load_initial_params()

        # --- Do everything on CPU ---
        with jax.default_device(jax.devices("cpu")[0]):

            # Turn 'a/b/c' -> ('a','b','c') and unflatten
            flat_dict = {}
            for k, v in loaded_params.items():
                # If k is already a tuple (like ('transformer', 'layer_12', 'mlp', 'linear', 'w')), just use it.
                if isinstance(k, tuple):
                    flat_dict[k] = v
                # Otherwise split only once.
                elif isinstance(k, str):
                    flat_dict[tuple(k.split('/'))] = v
                else:
                    raise TypeError(f"Unexpected key type: {type(k)} for key {k}")
            flat_original = flax.traverse_util.unflatten_dict(flat_dict)

            flat_remapped = {}
            weights_to_stack = defaultdict(list)
           

            print("Remapping checkpoint keys to match the nn.scan model structure...")

            transformer = flat_original.get('transformer', {})
            layer_pattern = re.compile(r'layer_(\d+)')

            weights_to_stack = defaultdict(list)
            layerless = {}

            # Separate per-layer and global transformer weights
            for key, value in transformer.items():
                m = layer_pattern.match(key)
                if m:
                    layer_idx = int(m.group(1))
                    for subkey, subval in flax.traverse_util.flatten_dict(value, sep='/').items():
                        weights_to_stack[subkey].append((layer_idx, subval))
                else:
                    layerless[key] = value  # e.g., final_norm, etc.

            # Stack all layer weights
            flat_remapped = {}
            for subkey, layer_values in weights_to_stack.items():
                layer_values.sort(key=lambda x: x[0])
                arrays = [v for _, v in layer_values]
                flat_remapped[('layers',) + tuple(subkey.split('/'))] = jnp.stack(arrays, axis=0)


            # Add layerless (non-layer) weights back
            for key, value in flax.traverse_util.flatten_dict(layerless, sep='/').items():
                #flat_remapped[('transformer',) + tuple(key.split('/'))] = value
                flat_remapped[('transformer',) + tuple(key.split('/'))] = jnp.array(value)
                
                
                

            # Rebuild final structure
            remapped_params = flax.traverse_util.unflatten_dict(flat_remapped)

            # Do final adjustments to naming
            if 'final_norm' in remapped_params.get('transformer', {}):
                remapped_params['final_norm'] = remapped_params['transformer'].pop('final_norm')

            if 'embedder' in remapped_params.get('transformer', {}):
                remapped_params['embedder'] = remapped_params['transformer'].pop('embedder')

            gemma3_keys = ['_key_norm', '_query_norm']
            pi0_keys = ['k_rmsnorm', 'q_rmsnorm']

            # Change remapped_params['layers'['attn']['_key_norm'] -> remapped_params['layers']['attn']['k_rmsnorm']
            # same for query
            remapped_params['layers']['attn']['k_rmsnorm'] = remapped_params['layers']['attn'].pop('_key_norm')
            remapped_params['layers']['attn']['q_rmsnorm'] = remapped_params['layers']['attn'].pop('_query_norm')

            # Fix Einsum in Gemma3
            gating_dict = remapped_params['layers']['mlp'].pop('gating_einsum')
            remapped_params['layers']['mlp']['Einsum_0'] = {}
            remapped_params['layers']['mlp']['Einsum_0']['gating_einsum'] = gating_dict['w']   

            linear_dict = remapped_params['layers']['mlp'].pop('linear')
             
            remapped_params['layers']['mlp']['Einsum_1'] = {}
            remapped_params['layers']['mlp']['Einsum_1']['linear'] = linear_dict['w']


            flat_model = flax.traverse_util.flatten_dict(params, sep='/')
            flat_ckpt = flax.traverse_util.flatten_dict(remapped_params, sep='/')

            matched = []
            unused = []
            missing = []

            print("Gemma Model Keys")
            for k in flat_ckpt.keys():
                print(f"  {k}")
               

            print("MoE Model Keys")
            for k in flat_model.keys():
                print(f"  {k}")
               

            print(f"Matched: {len(matched)}  Unused: {len(unused)}  Missing: {len(missing)}")

            # --- Merge into model params ---
            merged = _merge_params(flat_ckpt, flat_model, missing_regex=".*") # first arg are parameters to load, second arg is reference parameters

            # Get siglip and multimodal embeddings from gemma3 and put into unmerged
            unmerged = {}
            unmerged['SigLiPFromPatches_0'] = flat_original['SigLiPFromPatches_0']
            unmerged['embedder'] = flat_original['embedder']
            
        return merged, unmerged
        


        # params[layers][attn][attn_vec_einsum][w] -> flat_original[transformer][layer_{x}][attn][attn_vec_einsum][w]
        # params[layers][attn][k_rmsnorm][scale] -> flat_original[transformer][layer_{x}][attn][_key_norm][scale]
        # params[layers][attn][kv_einsum][w] -> flat_original[transformer][layer_{x}][attn][kv_einsum][w]
        # params[layers][attn][q_einsum][w] -> flat_original[transformer][layer_{x}][attn][q_einsum][w]
        # params[layers][attn][q_rmsnorm][scale] -> flat_original[transformer][layer_{x}][attn][_query_norm][scale]

        # params[layers][mlp][Einsum_0][gating_einsum][w] -> flat_original[transformer][layer_{x}][mlp][gating_einsum][w]
        # params[layers][mlp][Einsum_1][linear][w] -> flat_original[transformer][layer_{x}][mlp][linear][w]

        # params[layers][pre_attention_norm][scale] -> flat_original[transformer][layer_{x}][pre_attention_norm][scale]
        # params[layers][post_attention_norm][scale] -> flat_original[transformer][layer_{x}][post_attention_norm][scale]

        # params[layers][pre_ffw_norm][scale] -> flat_original[transformer][layer_{x}][pre_ffw_norm][scale]
        # params[layers][post_ffw_norm][scale] -> flat_original[transformer][layer_{x}][post_ffw_norm][scale]

        # params[final_norm][scale] -> flat_original[transformer][final_norm][scale]





        

        

# --- Step 2: Define model config (same as before) ---
VARIANT = "gemma3_4b"
action_expert = "gemma3_300m"
print(f"Setting up model for variant: {VARIANT} and action expert: {action_expert}")
config_gemma3 = gemma3.get_config(VARIANT)
#config_action_expert = gemma3.get_config(action_expert)
configs = [config_gemma3]

# --- Step 3: Instantiate the UNMODIFIED Model ---
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
loader = Gemma3ScanCompatibleLoader(params_path=gemma3_weights_path)

loaded_params = loader.load(params=initialized_model_params)
print("✅ Gemma 3 weights loaded successfully using the key-matching adapter!")