#!/usr/bin/env python3
"""
Convert a PyTorch PaliGemma model to JAX format.

This script loads a PyTorch PaliGemma checkpoint (from HuggingFace format) and converts
it to the JAX/Flax format used by the OpenPI codebase.

Usage:
    python convert_pytorch_model_to_jax.py --checkpoint_dir /path/to/pytorch/model --output_path /path/to/jax/checkpoint

Example:
    python convert_pytorch_model_to_jax.py --checkpoint_dir ./paligemma2-3b-pt-224 --output_path ./paligemma2-3b-jax
"""

import json
import os
import pathlib
from typing import Any

import numpy as np
import safetensors.torch
import torch
import tyro


def load_pytorch_model(checkpoint_dir: str) -> dict[str, torch.Tensor]:
    """Load PyTorch model from safetensors files."""
    checkpoint_path = pathlib.Path(checkpoint_dir)

    # Load the index file to find all weight files
    index_file = checkpoint_path / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            index_data = json.load(f)

        # Load all safetensors files
        state_dict = {}
        weight_files = set(index_data["weight_map"].values())
        for weight_file in weight_files:
            file_path = checkpoint_path / weight_file
            with safetensors.torch.safe_open(file_path, framework="pt") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
    else:
        # Single file model
        model_file = checkpoint_path / "model.safetensors"
        with safetensors.torch.safe_open(model_file, framework="pt") as f:
            state_dict = {key: f.get_tensor(key) for key in f.keys()}

    return state_dict


def convert_vision_tower_to_jax(state_dict: dict[str, torch.Tensor], config: dict[str, Any]) -> dict[str, np.ndarray]:
    """Convert vision tower weights from PyTorch to JAX format."""
    jax_params = {}
    num_layers = config["vision_config"]["num_hidden_layers"]
    hidden_size = config["vision_config"]["hidden_size"]
    num_heads = config["vision_config"]["num_attention_heads"]

    # Helper function to convert tensor to numpy (handles bfloat16)
    def to_numpy(tensor):
        if tensor.dtype == torch.bfloat16:
            return tensor.float().cpu().numpy().astype(np.float32)
        return tensor.cpu().numpy()

    # Patch embedding: Conv2d weight needs OIHW -> HWIO transpose
    patch_weight = state_dict["vision_tower.vision_model.embeddings.patch_embedding.weight"]
    jax_params["img/embedding/kernel"] = to_numpy(patch_weight.permute(2, 3, 1, 0))

    # Patch embedding bias
    patch_bias = state_dict["vision_tower.vision_model.embeddings.patch_embedding.bias"]
    jax_params["img/embedding/bias"] = to_numpy(patch_bias)

    # Position embedding: reshape to original format
    pos_emb = state_dict["vision_tower.vision_model.embeddings.position_embedding.weight"]
    # PyTorch shape: [num_patches, hidden_size], JAX expects [1, num_patches, hidden_size]
    jax_params["img/pos_embedding"] = to_numpy(pos_emb)[None, :, :]

    # Stack encoder layer parameters across all layers
    # Collect parameters for all layers
    layernorm0_scale = []
    layernorm0_bias = []
    layernorm1_scale = []
    layernorm1_bias = []

    mlp_dense0_kernel = []
    mlp_dense0_bias = []
    mlp_dense1_kernel = []
    mlp_dense1_bias = []

    attn_q_kernel = []
    attn_q_bias = []
    attn_k_kernel = []
    attn_k_bias = []
    attn_v_kernel = []
    attn_v_bias = []
    attn_out_kernel = []
    attn_out_bias = []

    for i in range(num_layers):
        prefix = f"vision_tower.vision_model.encoder.layers.{i}"

        # Layer norms
        layernorm0_scale.append(to_numpy(state_dict[f"{prefix}.layer_norm1.weight"]))
        layernorm0_bias.append(to_numpy(state_dict[f"{prefix}.layer_norm1.bias"]))
        layernorm1_scale.append(to_numpy(state_dict[f"{prefix}.layer_norm2.weight"]))
        layernorm1_bias.append(to_numpy(state_dict[f"{prefix}.layer_norm2.bias"]))

        # MLP - transpose weights from [out, in] to [in, out]
        mlp_dense0_kernel.append(to_numpy(state_dict[f"{prefix}.mlp.fc1.weight"].T))
        mlp_dense0_bias.append(to_numpy(state_dict[f"{prefix}.mlp.fc1.bias"]))
        mlp_dense1_kernel.append(to_numpy(state_dict[f"{prefix}.mlp.fc2.weight"].T))
        mlp_dense1_bias.append(to_numpy(state_dict[f"{prefix}.mlp.fc2.bias"]))

        # Attention - transpose and reshape
        # PyTorch stores as [out_features, in_features], JAX needs [in, num_heads, head_dim]
        q_weight = to_numpy(state_dict[f"{prefix}.self_attn.q_proj.weight"].T)
        q_weight_reshaped = q_weight.reshape(hidden_size, num_heads, hidden_size // num_heads)
        attn_q_kernel.append(q_weight_reshaped)

        q_bias = to_numpy(state_dict[f"{prefix}.self_attn.q_proj.bias"])
        q_bias_reshaped = q_bias.reshape(num_heads, hidden_size // num_heads)
        attn_q_bias.append(q_bias_reshaped)

        k_weight = to_numpy(state_dict[f"{prefix}.self_attn.k_proj.weight"].T)
        k_weight_reshaped = k_weight.reshape(hidden_size, num_heads, hidden_size // num_heads)
        attn_k_kernel.append(k_weight_reshaped)

        k_bias = to_numpy(state_dict[f"{prefix}.self_attn.k_proj.bias"])
        k_bias_reshaped = k_bias.reshape(num_heads, hidden_size // num_heads)
        attn_k_bias.append(k_bias_reshaped)

        v_weight = to_numpy(state_dict[f"{prefix}.self_attn.v_proj.weight"].T)
        v_weight_reshaped = v_weight.reshape(hidden_size, num_heads, hidden_size // num_heads)
        attn_v_kernel.append(v_weight_reshaped)

        v_bias = to_numpy(state_dict[f"{prefix}.self_attn.v_proj.bias"])
        v_bias_reshaped = v_bias.reshape(num_heads, hidden_size // num_heads)
        attn_v_bias.append(v_bias_reshaped)

        out_weight = to_numpy(state_dict[f"{prefix}.self_attn.out_proj.weight"].T)
        out_weight_reshaped = out_weight.reshape(num_heads, hidden_size // num_heads, hidden_size)
        attn_out_kernel.append(out_weight_reshaped)

        out_bias = to_numpy(state_dict[f"{prefix}.self_attn.out_proj.bias"])
        # out_bias should NOT be reshaped - keep as [hidden_size]
        attn_out_bias.append(out_bias)

    # Stack all layers (layer dimension first)
    jax_params["img/Transformer/encoderblock/LayerNorm_0/scale"] = np.stack(layernorm0_scale, axis=0)
    jax_params["img/Transformer/encoderblock/LayerNorm_0/bias"] = np.stack(layernorm0_bias, axis=0)
    jax_params["img/Transformer/encoderblock/LayerNorm_1/scale"] = np.stack(layernorm1_scale, axis=0)
    jax_params["img/Transformer/encoderblock/LayerNorm_1/bias"] = np.stack(layernorm1_bias, axis=0)

    jax_params["img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel"] = np.stack(mlp_dense0_kernel, axis=0)
    jax_params["img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias"] = np.stack(mlp_dense0_bias, axis=0)
    jax_params["img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel"] = np.stack(mlp_dense1_kernel, axis=0)
    jax_params["img/Transformer/encoderblock/MlpBlock_0/Dense_1/bias"] = np.stack(mlp_dense1_bias, axis=0)

    jax_params["img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/kernel"] = np.stack(
        attn_q_kernel, axis=0
    )
    jax_params["img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/bias"] = np.stack(attn_q_bias, axis=0)
    jax_params["img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/kernel"] = np.stack(
        attn_k_kernel, axis=0
    )
    jax_params["img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/bias"] = np.stack(attn_k_bias, axis=0)
    jax_params["img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/kernel"] = np.stack(
        attn_v_kernel, axis=0
    )
    jax_params["img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/bias"] = np.stack(attn_v_bias, axis=0)
    jax_params["img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel"] = np.stack(
        attn_out_kernel, axis=0
    )
    jax_params["img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/bias"] = np.stack(attn_out_bias, axis=0)

    # Post layer norm
    post_ln_weight = state_dict["vision_tower.vision_model.post_layernorm.weight"]
    jax_params["img/Transformer/encoder_norm/scale"] = to_numpy(post_ln_weight)

    post_ln_bias = state_dict["vision_tower.vision_model.post_layernorm.bias"]
    jax_params["img/Transformer/encoder_norm/bias"] = to_numpy(post_ln_bias)

    return jax_params


def convert_multimodal_projector_to_jax(state_dict: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Convert multi-modal projector weights from PyTorch to JAX format."""
    jax_params = {}

    # Helper function to convert tensor to numpy (handles bfloat16)
    def to_numpy(tensor):
        if tensor.dtype == torch.bfloat16:
            return tensor.float().cpu().numpy().astype(np.float32)
        return tensor.cpu().numpy()

    # Linear projection - transpose weight
    weight = state_dict["multi_modal_projector.linear.weight"]
    jax_params["img/head/kernel"] = to_numpy(weight.T)

    bias = state_dict["multi_modal_projector.linear.bias"]
    jax_params["img/head/bias"] = to_numpy(bias)

    return jax_params


def convert_language_model_to_jax(state_dict: dict[str, torch.Tensor], config: dict[str, Any]) -> dict[str, np.ndarray]:
    """Convert language model (Gemma2) weights from PyTorch to JAX format."""
    jax_params = {}
    text_config = config["text_config"]
    num_layers = text_config["num_hidden_layers"]
    hidden_size = text_config["hidden_size"]
    num_heads = text_config["num_attention_heads"]
    head_dim = hidden_size // num_heads
    num_kv_heads = text_config.get("num_key_value_heads", num_heads)

    print(
        f"  Language model: {num_layers} layers, hidden_size={hidden_size}, "
        f"num_heads={num_heads}, head_dim={head_dim}, num_kv_heads={num_kv_heads}"
    )

    # Helper function to convert tensor to numpy (handles bfloat16)
    def to_numpy(tensor):
        if tensor.dtype == torch.bfloat16:
            return tensor.float().cpu().numpy().astype(np.float32)
        return tensor.cpu().numpy()

    # Embedding table
    emb_weight = state_dict["language_model.model.embed_tokens.weight"]
    jax_params["llm/embedder/input_embedding"] = to_numpy(emb_weight)

    # Collect layer parameters for stacking
    q_einsum_list = []
    kv_einsum_list = []
    attn_vec_einsum_list = []
    gating_einsum_list = []
    linear_list = []
    input_layernorm_list = []
    post_attention_layernorm_list = []

    for i in range(num_layers):
        prefix = f"language_model.model.layers.{i}"

        # Attention Q projection
        # PyTorch shape: [num_heads * head_dim, hidden_size]
        # JAX expects: [num_heads, hidden_size, head_dim]
        q_weight = to_numpy(state_dict[f"{prefix}.self_attn.q_proj.weight"])
        q_out_dim, q_in_dim = q_weight.shape
        actual_head_dim = q_out_dim // num_heads

        # Reshape: [num_heads * head_dim, hidden_size] -> [num_heads, head_dim, hidden_size] -> [num_heads, hidden_size, head_dim]
        q_reshaped = q_weight.reshape(num_heads, actual_head_dim, hidden_size).transpose(0, 2, 1)
        q_einsum_list.append(q_reshaped)

        # K and V projections for multi-query/grouped-query attention
        # PyTorch shape: [num_kv_heads * head_dim, hidden_size]
        # JAX expects: [2, num_kv_heads, hidden_size, head_dim] for the stacked K/V
        k_weight = to_numpy(state_dict[f"{prefix}.self_attn.k_proj.weight"])
        v_weight = to_numpy(state_dict[f"{prefix}.self_attn.v_proj.weight"])

        # Get actual dimensions from K/V weights
        kv_out_dim = k_weight.shape[0]
        kv_head_dim = kv_out_dim // num_kv_heads

        # Transpose and reshape K: [num_kv_heads * head_dim, hidden_size] -> [num_kv_heads, hidden_size, head_dim]
        k_reshaped = k_weight.T.reshape(hidden_size, num_kv_heads, kv_head_dim).transpose(1, 0, 2)
        # Transpose and reshape V: [num_kv_heads * head_dim, hidden_size] -> [num_kv_heads, hidden_size, head_dim]
        v_reshaped = v_weight.T.reshape(hidden_size, num_kv_heads, kv_head_dim).transpose(1, 0, 2)

        # Stack K and V: [2, num_kv_heads, hidden_size, head_dim]
        kv_stacked = np.stack([k_reshaped, v_reshaped], axis=0)
        kv_einsum_list.append(kv_stacked)

        # O projection: [hidden_size, num_heads * head_dim] -> [num_heads, head_dim, hidden_size]
        o_weight = to_numpy(state_dict[f"{prefix}.self_attn.o_proj.weight"])
        o_reshaped = o_weight.T.reshape(num_heads, actual_head_dim, hidden_size)
        attn_vec_einsum_list.append(o_reshaped)

        # MLP gate and up projections
        gate_weight = to_numpy(state_dict[f"{prefix}.mlp.gate_proj.weight"].T)
        up_weight = to_numpy(state_dict[f"{prefix}.mlp.up_proj.weight"].T)
        # Stack gate and up: [2, hidden_size, intermediate_size]
        gating_stacked = np.stack([gate_weight, up_weight], axis=0)
        gating_einsum_list.append(gating_stacked)

        # MLP down projection
        down_weight = to_numpy(state_dict[f"{prefix}.mlp.down_proj.weight"].T)
        linear_list.append(down_weight)

        # Layer norms
        input_ln = to_numpy(state_dict[f"{prefix}.input_layernorm.weight"])
        input_layernorm_list.append(input_ln)

        post_attn_ln = to_numpy(state_dict[f"{prefix}.post_attention_layernorm.weight"])
        post_attention_layernorm_list.append(post_attn_ln)

    # Stack all layer parameters
    jax_params["llm/layers/attn/q_einsum/w"] = np.stack(q_einsum_list, axis=0)
    jax_params["llm/layers/attn/kv_einsum/w"] = np.stack(kv_einsum_list, axis=0)
    jax_params["llm/layers/attn/attn_vec_einsum/w"] = np.stack(attn_vec_einsum_list, axis=0)
    jax_params["llm/layers/mlp/gating_einsum"] = np.stack(gating_einsum_list, axis=0)
    jax_params["llm/layers/mlp/linear"] = np.stack(linear_list, axis=0)
    jax_params["llm/layers/pre_attention_norm/scale"] = np.stack(input_layernorm_list, axis=0)
    jax_params["llm/layers/pre_ffw_norm/scale"] = np.stack(post_attention_layernorm_list, axis=0)

    # Final layer norm
    final_ln = state_dict["language_model.model.norm.weight"]
    jax_params["llm/final_norm/scale"] = to_numpy(final_ln)

    return jax_params


def save_jax_checkpoint(params: dict[str, np.ndarray], output_path: str):
    """Save parameters as .npz file (uncompressed for speed)."""
    os.makedirs(output_path, exist_ok=True)

    # The loader expects parameters with a "params/" prefix after unflattening
    # So we need to add "params/" to all keys
    params_with_prefix = {f"params/{k}": v for k, v in params.items()}

    # Save as uncompressed npz file for much faster saving
    # File size will be larger but saving is 10-100x faster
    save_path = os.path.join(output_path, "params", "params.npz")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Saving {len(params_with_prefix)} parameters to {save_path}...")
    np.savez(save_path, **params_with_prefix)  # Use np.savez instead of np.savez_compressed

    print(f"Checkpoint saved to {save_path}")

    # Also save parameter metadata as JSON for reference
    metadata = {
        "parameter_keys": list(params.keys()),
        "parameter_shapes": {k: v.shape for k, v in params.items()},
        "parameter_dtypes": {k: str(v.dtype) for k, v in params.items()},
    }

    metadata_path = os.path.join(output_path, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"Metadata saved to {metadata_path}")


def main(
    checkpoint_dir: str,
    output_path: str,
):
    """Convert PyTorch PaliGemma model to JAX format.

    Args:
        checkpoint_dir: Path to the PyTorch checkpoint directory
        output_path: Path to save the JAX checkpoint
    """
    print(f"Loading PyTorch model from {checkpoint_dir}")

    # Load config
    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    print(f"Model config: {config['model_type']}")
    print(f"  Vision layers: {config['vision_config']['num_hidden_layers']}")
    print(f"  Text layers: {config['text_config']['num_hidden_layers']}")

    # Load PyTorch model
    state_dict = load_pytorch_model(checkpoint_dir)
    print(f"Loaded {len(state_dict)} parameters from PyTorch model")

    # Convert each component
    print("Converting vision tower...")
    vision_params = convert_vision_tower_to_jax(state_dict, config)

    print("Converting multi-modal projector...")
    projector_params = convert_multimodal_projector_to_jax(state_dict)

    print("Converting language model...")
    language_params = convert_language_model_to_jax(state_dict, config)

    # Combine all parameters
    all_params = {**vision_params, **projector_params, **language_params}

    print(f"Total JAX parameters: {len(all_params)}")

    # Print shape verification
    print("\nParameter shapes verification:")
    print("Vision Tower:")
    for key in sorted([k for k in all_params if k.startswith("img/")]):
        print(f"  {key}: {all_params[key].shape}")
    print("\nLanguage Model (sample):")
    for key in sorted([k for k in all_params if k.startswith("llm/")])[:5]:
        print(f"  {key}: {all_params[key].shape}")

    # Save checkpoint
    print(f"\nSaving JAX checkpoint to {output_path}")
    save_jax_checkpoint(all_params, output_path)

    print("\nConversion completed successfully!")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    tyro.cli(main)
