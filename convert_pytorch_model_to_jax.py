#!/usr/bin/env python3
"""
Convert PyTorch PaliGemma model to JAX format.

This script loads a PyTorch PaliGemma model and converts it to JAX format
using the same structure as the original JAX models.

Usage:
    # Convert PyTorch model to JAX:
    python convert_pytorch_model_to_jax.py --pytorch_model_path /path/to/pytorch/model --output_path /path/to/output

Example:
    python convert_pytorch_model_to_jax.py --pytorch_model_path paligemma2-3b-pt-224 --output_path paligemma2-3b-jax
"""

import argparse
import json
import os
import shutil
from typing import Any

import jax.numpy as jnp
import numpy as np
import safetensors
import torch


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert PyTorch tensor to numpy array, handling BFloat16 conversion."""
    if tensor.dtype == torch.bfloat16:
        return tensor.float().numpy()
    return tensor.numpy()


def load_pytorch_model(pytorch_model_path: str) -> dict[str, torch.Tensor]:
    """Load PyTorch model from safetensors files."""
    model_files = []
    for file in os.listdir(pytorch_model_path):
        if file.endswith(".safetensors") and file.startswith("model-"):
            model_files.append(os.path.join(pytorch_model_path, file))

    model_files.sort()  # Ensure correct order

    state_dict = {}
    for model_file in model_files:
        with safetensors.safe_open(model_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    return state_dict


def convert_vision_tower_to_jax(state_dict: dict[str, torch.Tensor], config: dict[str, Any]) -> dict[str, jnp.ndarray]:
    """Convert vision tower from PyTorch to JAX format."""
    jax_params = {}

    # Patch embeddings
    pytorch_key = "vision_tower.vision_model.embeddings.patch_embedding.weight"
    if pytorch_key in state_dict:
        jax_key = "img/embedding/kernel"
        # Convert from (out_channels, in_channels, kernel_height, kernel_width) to (kernel_height, kernel_width, in_channels, out_channels)
        # Convert to float32 first to handle BFloat16
        weight = tensor_to_numpy(state_dict[pytorch_key]).transpose(2, 3, 1, 0)
        jax_params[jax_key] = jnp.array(weight)

    pytorch_key = "vision_tower.vision_model.embeddings.patch_embedding.bias"
    if pytorch_key in state_dict:
        jax_key = "img/embedding/bias"
        jax_params[jax_key] = jnp.array(tensor_to_numpy(state_dict[pytorch_key]))

    # Positional embeddings
    pytorch_key = "vision_tower.vision_model.embeddings.position_embedding.weight"
    if pytorch_key in state_dict:
        jax_key = "img/pos_embedding"
        # Reshape from (num_positions, hidden_size) to (height, width, hidden_size)
        pos_emb = tensor_to_numpy(state_dict[pytorch_key])
        height = int(np.sqrt(pos_emb.shape[0]))
        width = height
        jax_params[jax_key] = jnp.array(pos_emb.reshape(height, width, -1))

    # Vision encoder layers
    num_layers = config["vision_config"]["num_hidden_layers"]

    # Collect layer norm parameters
    layernorm0_scales = []
    layernorm0_biases = []
    layernorm1_scales = []
    layernorm1_biases = []

    # Collect MLP parameters
    mlp_dense0_kernels = []
    mlp_dense0_biases = []
    mlp_dense1_kernels = []
    mlp_dense1_biases = []

    # Collect attention parameters - these will be stored as single arrays per attention type
    attn_key_kernels = []
    attn_key_biases = []
    attn_value_kernels = []
    attn_value_biases = []
    attn_query_kernels = []
    attn_query_biases = []
    attn_out_kernels = []
    attn_out_biases = []

    for i in range(num_layers):
        # Layer norms
        pytorch_key = f"vision_tower.vision_model.encoder.layers.{i}.layer_norm1.weight"
        if pytorch_key in state_dict:
            layernorm0_scales.append(tensor_to_numpy(state_dict[pytorch_key]).T)

        pytorch_key = f"vision_tower.vision_model.encoder.layers.{i}.layer_norm1.bias"
        if pytorch_key in state_dict:
            layernorm0_biases.append(tensor_to_numpy(state_dict[pytorch_key]))

        pytorch_key = f"vision_tower.vision_model.encoder.layers.{i}.layer_norm2.weight"
        if pytorch_key in state_dict:
            layernorm1_scales.append(tensor_to_numpy(state_dict[pytorch_key]).T)

        pytorch_key = f"vision_tower.vision_model.encoder.layers.{i}.layer_norm2.bias"
        if pytorch_key in state_dict:
            layernorm1_biases.append(tensor_to_numpy(state_dict[pytorch_key]))

        # MLP layers
        pytorch_key = f"vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.weight"
        if pytorch_key in state_dict:
            mlp_dense0_kernels.append(tensor_to_numpy(state_dict[pytorch_key]).T)

        pytorch_key = f"vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.bias"
        if pytorch_key in state_dict:
            mlp_dense0_biases.append(tensor_to_numpy(state_dict[pytorch_key]))

        pytorch_key = f"vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.weight"
        if pytorch_key in state_dict:
            mlp_dense1_kernels.append(tensor_to_numpy(state_dict[pytorch_key]).T)

        pytorch_key = f"vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.bias"
        if pytorch_key in state_dict:
            mlp_dense1_biases.append(tensor_to_numpy(state_dict[pytorch_key]))

        # Attention layers - collect per layer, will reshape later
        pytorch_key = f"vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.weight"
        if pytorch_key in state_dict:
            weight = tensor_to_numpy(state_dict[pytorch_key]).T
            num_heads = config["vision_config"]["num_attention_heads"]
            head_dim = weight.shape[0] // num_heads
            # Store as (hidden_size, num_heads, head_dim) for later reshaping
            attn_key_kernels.append(weight.reshape(-1, num_heads, head_dim))

        pytorch_key = f"vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.bias"
        if pytorch_key in state_dict:
            bias = tensor_to_numpy(state_dict[pytorch_key])
            num_heads = config["vision_config"]["num_attention_heads"]
            head_dim = bias.shape[0] // num_heads
            attn_key_biases.append(bias.reshape(num_heads, head_dim))

        pytorch_key = f"vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.weight"
        if pytorch_key in state_dict:
            weight = tensor_to_numpy(state_dict[pytorch_key]).T
            num_heads = config["vision_config"]["num_attention_heads"]
            head_dim = weight.shape[0] // num_heads
            attn_value_kernels.append(weight.reshape(-1, num_heads, head_dim))

        pytorch_key = f"vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.bias"
        if pytorch_key in state_dict:
            bias = tensor_to_numpy(state_dict[pytorch_key])
            num_heads = config["vision_config"]["num_attention_heads"]
            head_dim = bias.shape[0] // num_heads
            attn_value_biases.append(bias.reshape(num_heads, head_dim))

        pytorch_key = f"vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.weight"
        if pytorch_key in state_dict:
            weight = tensor_to_numpy(state_dict[pytorch_key]).T
            num_heads = config["vision_config"]["num_attention_heads"]
            head_dim = weight.shape[0] // num_heads
            attn_query_kernels.append(weight.reshape(-1, num_heads, head_dim))

        pytorch_key = f"vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.bias"
        if pytorch_key in state_dict:
            bias = tensor_to_numpy(state_dict[pytorch_key])
            num_heads = config["vision_config"]["num_attention_heads"]
            head_dim = bias.shape[0] // num_heads
            attn_query_biases.append(bias.reshape(num_heads, head_dim))

        pytorch_key = f"vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.weight"
        if pytorch_key in state_dict:
            weight = tensor_to_numpy(state_dict[pytorch_key]).T
            num_heads = config["vision_config"]["num_attention_heads"]
            head_dim = weight.shape[0] // num_heads
            attn_out_kernels.append(weight.reshape(-1, num_heads, head_dim))

        pytorch_key = f"vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.bias"
        if pytorch_key in state_dict:
            bias = tensor_to_numpy(state_dict[pytorch_key])
            num_heads = config["vision_config"]["num_attention_heads"]
            head_dim = bias.shape[0] // num_heads
            attn_out_biases.append(bias.reshape(num_heads, head_dim))

    # Store collected parameters in JAX format
    if layernorm0_scales:
        jax_params["img/Transformer/encoderblock/LayerNorm_0/scale"] = jnp.array(layernorm0_scales)
    if layernorm0_biases:
        jax_params["img/Transformer/encoderblock/LayerNorm_0/bias"] = jnp.array(layernorm0_biases)
    if layernorm1_scales:
        jax_params["img/Transformer/encoderblock/LayerNorm_1/scale"] = jnp.array(layernorm1_scales)
    if layernorm1_biases:
        jax_params["img/Transformer/encoderblock/LayerNorm_1/bias"] = jnp.array(layernorm1_biases)

    if mlp_dense0_kernels:
        jax_params["img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel"] = jnp.array(mlp_dense0_kernels)
    if mlp_dense0_biases:
        jax_params["img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias"] = jnp.array(mlp_dense0_biases)
    if mlp_dense1_kernels:
        jax_params["img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel"] = jnp.array(mlp_dense1_kernels)
    if mlp_dense1_biases:
        jax_params["img/Transformer/encoderblock/MlpBlock_0/Dense_1/bias"] = jnp.array(mlp_dense1_biases)

    if attn_key_kernels:
        # Convert from list of (hidden_size, num_heads, head_dim) to (num_layers, hidden_size, num_heads, head_dim)
        jax_params["img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/kernel"] = jnp.array(
            attn_key_kernels
        )
    if attn_key_biases:
        # Convert from list of (num_heads, head_dim) to (num_layers, num_heads, head_dim)
        jax_params["img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/bias"] = jnp.array(attn_key_biases)
    if attn_value_kernels:
        jax_params["img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/kernel"] = jnp.array(
            attn_value_kernels
        )
    if attn_value_biases:
        jax_params["img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/bias"] = jnp.array(
            attn_value_biases
        )
    if attn_query_kernels:
        jax_params["img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/kernel"] = jnp.array(
            attn_query_kernels
        )
    if attn_query_biases:
        jax_params["img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/bias"] = jnp.array(
            attn_query_biases
        )
    if attn_out_kernels:
        jax_params["img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel"] = jnp.array(
            attn_out_kernels
        )
    if attn_out_biases:
        jax_params["img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/bias"] = jnp.array(attn_out_biases)

    # Final layer norm
    pytorch_key = "vision_tower.vision_model.post_layernorm.weight"
    if pytorch_key in state_dict:
        jax_key = "img/Transformer/encoder_norm/scale"
        jax_params[jax_key] = jnp.array(tensor_to_numpy(state_dict[pytorch_key]).T)

    pytorch_key = "vision_tower.vision_model.post_layernorm.bias"
    if pytorch_key in state_dict:
        jax_key = "img/Transformer/encoder_norm/bias"
        jax_params[jax_key] = jnp.array(tensor_to_numpy(state_dict[pytorch_key]))

    return jax_params


def convert_multimodal_projector_to_jax(state_dict: dict[str, torch.Tensor]) -> dict[str, jnp.ndarray]:
    """Convert multimodal projector from PyTorch to JAX format."""
    jax_params = {}

    pytorch_key = "multi_modal_projector.linear.weight"
    if pytorch_key in state_dict:
        jax_key = "img/head/kernel"
        jax_params[jax_key] = jnp.array(tensor_to_numpy(state_dict[pytorch_key]).T)

    pytorch_key = "multi_modal_projector.linear.bias"
    if pytorch_key in state_dict:
        jax_key = "img/head/bias"
        jax_params[jax_key] = jnp.array(tensor_to_numpy(state_dict[pytorch_key]))

    return jax_params


def convert_language_model_to_jax(
    state_dict: dict[str, torch.Tensor], config: dict[str, Any]
) -> dict[str, jnp.ndarray]:
    """Convert language model from PyTorch to JAX format."""
    jax_params = {}

    # Embedding layer
    pytorch_key = "language_model.model.embed_tokens.weight"
    if pytorch_key in state_dict:
        jax_key = "llm/embedder/input_embedding"
        jax_params[jax_key] = jnp.array(tensor_to_numpy(state_dict[pytorch_key]))

    num_layers = config["text_config"]["num_hidden_layers"]
    num_heads = config["text_config"]["num_attention_heads"]
    head_dim = config["text_config"]["hidden_size"] // num_heads

    # Collect attention parameters
    attn_vec_einsum = []
    kv_einsum = []
    q_einsum = []

    # Collect MLP parameters
    gating_einsum = []
    linear_mlp = []

    # Collect layer norm parameters
    input_layernorm = []
    post_attention_layernorm = []

    for i in range(num_layers):
        # Attention weights
        pytorch_key = f"language_model.model.layers.{i}.self_attn.q_proj.weight"
        if pytorch_key in state_dict:
            weight = tensor_to_numpy(state_dict[pytorch_key])
            # Reshape from (hidden_size, num_heads * head_dim) to (num_heads, head_dim, hidden_size)
            q_weight = weight.T.reshape(num_heads, head_dim, -1)
            q_einsum.append(q_weight)

        pytorch_key = f"language_model.model.layers.{i}.self_attn.k_proj.weight"
        if pytorch_key in state_dict:
            weight = tensor_to_numpy(state_dict[pytorch_key])
            k_weight = weight.T.reshape(num_heads, head_dim, -1)
            kv_einsum.append([k_weight, np.zeros_like(k_weight)])  # Placeholder for v

        pytorch_key = f"language_model.model.layers.{i}.self_attn.v_proj.weight"
        if pytorch_key in state_dict:
            weight = tensor_to_numpy(state_dict[pytorch_key])
            v_weight = weight.T.reshape(num_heads, head_dim, -1)
            if i < len(kv_einsum):
                kv_einsum[i][1] = v_weight

        pytorch_key = f"language_model.model.layers.{i}.self_attn.o_proj.weight"
        if pytorch_key in state_dict:
            weight = tensor_to_numpy(state_dict[pytorch_key])
            # Reshape from (num_heads * head_dim, hidden_size) to (num_heads, head_dim, hidden_size)
            o_weight = weight.T.reshape(num_heads, head_dim, -1)
            attn_vec_einsum.append(o_weight)

        # MLP weights
        pytorch_key = f"language_model.model.layers.{i}.mlp.gate_proj.weight"
        if pytorch_key in state_dict:
            gate_weight = tensor_to_numpy(state_dict[pytorch_key]).T
            gating_einsum.append([gate_weight, np.zeros_like(gate_weight)])  # Placeholder for up

        pytorch_key = f"language_model.model.layers.{i}.mlp.up_proj.weight"
        if pytorch_key in state_dict:
            up_weight = tensor_to_numpy(state_dict[pytorch_key]).T
            if i < len(gating_einsum):
                gating_einsum[i][1] = up_weight

        pytorch_key = f"language_model.model.layers.{i}.mlp.down_proj.weight"
        if pytorch_key in state_dict:
            down_weight = tensor_to_numpy(state_dict[pytorch_key]).T
            linear_mlp.append(down_weight)

        # Layer norms
        pytorch_key = f"language_model.model.layers.{i}.input_layernorm.weight"
        if pytorch_key in state_dict:
            input_layernorm.append(tensor_to_numpy(state_dict[pytorch_key]))

        pytorch_key = f"language_model.model.layers.{i}.post_attention_layernorm.weight"
        if pytorch_key in state_dict:
            post_attention_layernorm.append(tensor_to_numpy(state_dict[pytorch_key]))

    # Store parameters in JAX format
    if attn_vec_einsum:
        jax_params["llm/layers/attn/attn_vec_einsum/w"] = jnp.array(attn_vec_einsum)
    if kv_einsum:
        jax_params["llm/layers/attn/kv_einsum/w"] = jnp.array(kv_einsum)
    if q_einsum:
        jax_params["llm/layers/attn/q_einsum/w"] = jnp.array(q_einsum)

    if gating_einsum:
        jax_params["llm/layers/mlp/gating_einsum"] = jnp.array(gating_einsum)
    if linear_mlp:
        jax_params["llm/layers/mlp/linear"] = jnp.array(linear_mlp)

    if input_layernorm:
        jax_params["llm/layers/pre_attention_norm/scale"] = jnp.array(input_layernorm)
    if post_attention_layernorm:
        jax_params["llm/layers/pre_ffw_norm/scale"] = jnp.array(post_attention_layernorm)

    # Final layer norm
    pytorch_key = "language_model.model.norm.weight"
    if pytorch_key in state_dict:
        jax_key = "llm/final_norm/scale"
        jax_params[jax_key] = jnp.array(tensor_to_numpy(state_dict[pytorch_key]))

    return jax_params


def convert_pytorch_to_jax(pytorch_model_path: str, output_path: str):
    """Convert PyTorch PaliGemma model to JAX format."""
    print(f"Converting PyTorch model from {pytorch_model_path} to {output_path}")

    # Load PyTorch model
    state_dict = load_pytorch_model(pytorch_model_path)
    print(f"Loaded {len(state_dict)} parameters from PyTorch model")

    # Load config
    config_path = os.path.join(pytorch_model_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    # Convert components
    print("Converting vision tower...")
    vision_params = convert_vision_tower_to_jax(state_dict, config)

    print("Converting multimodal projector...")
    projector_params = convert_multimodal_projector_to_jax(state_dict)

    print("Converting language model...")
    language_params = convert_language_model_to_jax(state_dict, config)

    # Combine all parameters
    all_params = {**vision_params, **projector_params, **language_params}

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Save parameters in JAX format
    params_path = os.path.join(output_path, "params")
    os.makedirs(params_path, exist_ok=True)

    # Convert to JAX format and save
    jax_params = {}
    for key, value in all_params.items():
        if isinstance(value, jnp.ndarray):
            jax_params[key] = value
        else:
            jax_params[key] = jnp.array(value)

    # Save as numpy arrays (simplified approach)
    np_params = {}
    for key, value in jax_params.items():
        np_params[key] = np.array(value)

    # Structure parameters so that when unflattened, they create the expected nested structure
    # The training code expects: {"PaliGemma": {"params": {...}}}
    # So we need to prefix all keys with "params/" so that unflattening creates the right structure
    structured_flat_params = {}
    for key, value in np_params.items():
        structured_flat_params[f"params/{key}"] = value

    # Save parameters as flat dictionary with "/" separators
    # The training code will unflatten this and access the "params" key
    np.savez(os.path.join(params_path, "params.npz"), **structured_flat_params)

    # Copy config files
    for config_file in [
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "preprocessor_config.json",
    ]:
        src_path = os.path.join(pytorch_model_path, config_file)
        if os.path.exists(src_path):
            shutil.copy2(src_path, os.path.join(output_path, config_file))

    print("Conversion completed successfully!")
    print(f"JAX model saved to {output_path}")
    print(f"Total parameters converted: {len(all_params)}")


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch PaliGemma model to JAX format")
    parser.add_argument("--pytorch_model_path", required=True, help="Path to PyTorch model directory")
    parser.add_argument("--output_path", required=True, help="Path to save JAX model")

    args = parser.parse_args()

    if not os.path.exists(args.pytorch_model_path):
        print(f"Error: PyTorch model path {args.pytorch_model_path} does not exist")
        return

    convert_pytorch_to_jax(args.pytorch_model_path, args.output_path)


if __name__ == "__main__":
    main()
