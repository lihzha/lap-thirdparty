#!/usr/bin/env python3
"""Visualize embodiment embeddings from trained models.

This script extracts embeddings from trained PiCoT models for different robot
embodiments and visualizes them to analyze cross-embodiment transfer.

Usage:
    python scripts/vis_embodiment_embeddings.py \
        --config pi_combined_cot_local \
        --exp_name my_exp \
        --checkpoint_step 10000 \
        --num_samples 500 \
        --embedding_type prefix_prelogits \
        --output_dir ./embeddings_viz

Embedding types:
    - image: SigLIP image embeddings (mean-pooled across patches)
    - prefix_prelogits: Pre-logits after prefix forward pass (combined image + text)
    - combined_prefix: Raw concatenated prefix embeddings before LLM
    - text_only: Text token embeddings only
    - per_camera: Separate embeddings for each camera view
"""

import argparse
import dataclasses
import json
import logging
import os
import platform
from collections import defaultdict
from pathlib import Path
from typing import Literal

import etils.epath as epath
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from openpi.models import model as _model
from openpi.training import optimizer as _optimizer
from rail_tpu_utils import prevent_cross_region
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tqdm

import openpi_cot.datasets.cot_data_loader as _data_loader
from openpi_cot.models.model_adapter import CoTObservation
import openpi_cot.training.checkpoints as _checkpoints
import openpi_cot.training.config as _config
import openpi_cot.training.mh_sharding as sharding
import openpi_cot.training.utils as training_utils

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


EmbeddingType = Literal[
    "image",
    "prefix_prelogits", 
    "combined_prefix",
    "text_only",
    "per_camera",
    "wrist_image",
]


@dataclasses.dataclass
class EmbeddingResult:
    """Container for extracted embeddings with metadata."""
    embeddings: np.ndarray  # [num_samples, embedding_dim]
    dataset_names: list[str]  # [num_samples]
    sample_indices: list[int]  # [num_samples] - for debugging/tracking
    

def init_tpu(config: _config.TrainConfig):
    """Initialize TPU/GPU runtime."""
    if (
        ("v6" in config.name and config.fsdp_devices > 8)
        or ("v4" in config.name and config.fsdp_devices > 4)
        or ("v5" in config.name and config.fsdp_devices > 4)
    ):
        jax.distributed.initialize()
    
    if "local" in config.name:
        os.environ["CURL_CA_BUNDLE"] = "/etc/pki/tls/certs/ca-bundle.crt"

    data_dir = config.data.rlds_data_dir
    
    def _is_tpu_runtime() -> bool:
        try:
            return any(d.platform == "tpu" for d in jax.devices())
        except Exception:
            return False
    
    if _is_tpu_runtime() and str(data_dir).startswith("gs://"):
        prevent_cross_region(data_dir, data_dir)
    
    # Determine effective FSDP devices
    process_count = jax.process_count()
    local_devices = jax.local_device_count()
    global_devices = jax.device_count()
    logger.info(f"Devices: local={local_devices}, global={global_devices}, processes={process_count}")
    
    if process_count == 1:
        target = min(config.fsdp_devices, local_devices)
        effective_fsdp = 1
        for d in range(target, 0, -1):
            if global_devices % d == 0:
                effective_fsdp = d
                break
    else:
        effective_fsdp = config.fsdp_devices
    
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))
    return effective_fsdp


def load_checkpoint(
    config: _config.TrainConfig,
    effective_fsdp: int,
    checkpoint_step: int | None = None,
) -> tuple[training_utils.TrainState, jax.sharding.Sharding]:
    """Load model checkpoint and return train state."""
    mesh = sharding.make_mesh(effective_fsdp)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    
    # Initialize checkpoint manager
    checkpoint_manager, _ = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=False,
        resume=True,
        async_enable=False,
    )
    
    # Create optimizer and training state
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)
    ema_decay, ema_params_enabled = config.get_ema_init()
    
    rng = jax.random.key(config.seed)
    
    def init(rng):
        model = config.model.create(rng)
        params = nnx.state(model)
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=ema_decay,
            ema_params=None if not ema_params_enabled else params,
        )
    
    train_state_shape = jax.eval_shape(init, rng)
    train_state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=False)
    
    # Determine checkpoint step
    available_steps = list(checkpoint_manager.all_steps())
    logger.info(f"Available checkpoints: {available_steps}")
    
    if checkpoint_step is None:
        checkpoint_step = checkpoint_manager.latest_step()
        logger.info(f"Using latest checkpoint: {checkpoint_step}")
    elif checkpoint_step not in available_steps:
        raise ValueError(f"Step {checkpoint_step} not found. Available: {available_steps}")
    
    # Restore checkpoint
    train_state = _checkpoints.restore_state(
        checkpoint_manager,
        train_state_shape,
        data_loader=None,
        step=checkpoint_step,
        train_state_sharding=train_state_sharding,
    )
    
    # Use EMA params if available
    if train_state.ema_params is not None:
        logger.info("Using EMA params")
        train_state = dataclasses.replace(train_state, params=train_state.ema_params)
    
    logger.info(f"Loaded checkpoint at step {train_state.step}")
    
    return train_state, data_sharding, mesh


class EmbeddingExtractor:
    """Extract embeddings from a PiCoT model."""
    
    def __init__(self, train_state: training_utils.TrainState):
        self.train_state = train_state
        self.model = nnx.merge(train_state.model_def, train_state.params)
        self.model.eval()
    
    def extract_image_embeddings(
        self,
        observation: CoTObservation,
        image_key: str = "base_0_rgb",
    ) -> jax.Array:
        """Extract mean-pooled SigLIP image embeddings.
        
        Args:
            observation: Model observation containing images
            image_key: Which image to extract embeddings from
            
        Returns:
            Embeddings of shape [batch, embedding_dim]
        """
        image = observation.images[image_key]
        # Normalize to [-1, 1] if not already
        if image.max() > 1.0:
            image = image / 127.5 - 1.0
        
        image_tokens, _ = self.model.PaliGemma.img(image, train=False)
        # Mean pool across patches: [b, num_patches, emb] -> [b, emb]
        return jnp.mean(image_tokens, axis=1)
    
    def extract_prefix_prelogits(
        self,
        observation: CoTObservation,
    ) -> jax.Array:
        """Extract pre-logits after full prefix forward pass.
        
        This captures the combined image + text representation after
        transformer processing.
        
        Args:
            observation: Model observation
            
        Returns:
            Embeddings of shape [batch, embedding_dim]
        """
        from openpi_cot.models.model_adapter import preprocess_observation
        
        # Preprocess observation (no augmentation for inference)
        obs = preprocess_observation(
            None,
            observation,
            train=False,
            image_keys=self.model.image_keys,
            aug_wrist_image=False,
        )
        
        # Get prefix embeddings
        prefix_tokens, prefix_mask, prefix_ar_mask = self.model.embed_prefix(obs)
        prefix_attn_mask = jnp.ones((prefix_tokens.shape[0], prefix_tokens.shape[1], prefix_tokens.shape[1]), dtype=bool)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        
        # Forward through LLM
        pre_logits, _ = self.model.PaliGemma.llm(
            [prefix_tokens],
            mask=prefix_attn_mask,
            positions=positions,
            adarms_cond=[None],
        )
        
        # Mean pool over valid tokens
        pre_logits_squeezed = pre_logits[0]  # [b, seq, emb]
        mask_expanded = prefix_mask[:, :, None].astype(jnp.float32)
        pooled = jnp.sum(pre_logits_squeezed * mask_expanded, axis=1) / jnp.maximum(
            jnp.sum(mask_expanded, axis=1), 1.0
        )
        return pooled
    
    def extract_combined_prefix(
        self,
        observation: CoTObservation,
    ) -> jax.Array:
        """Extract raw prefix embeddings before LLM processing.
        
        Args:
            observation: Model observation
            
        Returns:
            Embeddings of shape [batch, embedding_dim]
        """
        from openpi_cot.models.model_adapter import preprocess_observation
        
        obs = preprocess_observation(
            None,
            observation,
            train=False,
            image_keys=self.model.image_keys,
            aug_wrist_image=False,
        )
        
        prefix_tokens, prefix_mask, _ = self.model.embed_prefix(obs)
        
        # Mean pool over valid tokens
        mask_expanded = prefix_mask[:, :, None].astype(jnp.float32)
        pooled = jnp.sum(prefix_tokens * mask_expanded, axis=1) / jnp.maximum(
            jnp.sum(mask_expanded, axis=1), 1.0
        )
        return pooled
    
    def extract_text_embeddings(
        self,
        observation: CoTObservation,
    ) -> jax.Array:
        """Extract text-only embeddings.
        
        Args:
            observation: Model observation
            
        Returns:
            Embeddings of shape [batch, embedding_dim]
        """
        # Embed text tokens
        text_embeddings = self.model.PaliGemma.llm(
            observation.tokenized_prompt, method="embed"
        )
        
        # Mean pool over valid text tokens
        mask = observation.tokenized_prompt_mask[:, :, None].astype(jnp.float32)
        pooled = jnp.sum(text_embeddings * mask, axis=1) / jnp.maximum(
            jnp.sum(mask, axis=1), 1.0
        )
        return pooled
    
    def extract(
        self,
        observation: CoTObservation,
        embedding_type: EmbeddingType,
    ) -> jax.Array:
        """Extract embeddings of the specified type.
        
        Args:
            observation: Model observation
            embedding_type: Type of embedding to extract
            
        Returns:
            Embeddings of shape [batch, embedding_dim]
        """
        if embedding_type == "image":
            return self.extract_image_embeddings(observation, "base_0_rgb")
        elif embedding_type == "wrist_image":
            return self.extract_image_embeddings(observation, "left_wrist_0_rgb")
        elif embedding_type == "prefix_prelogits":
            return self.extract_prefix_prelogits(observation)
        elif embedding_type == "combined_prefix":
            return self.extract_combined_prefix(observation)
        elif embedding_type == "text_only":
            return self.extract_text_embeddings(observation)
        elif embedding_type == "per_camera":
            # Concatenate base and wrist image embeddings
            base_emb = self.extract_image_embeddings(observation, "base_0_rgb")
            wrist_emb = self.extract_image_embeddings(observation, "left_wrist_0_rgb")
            return jnp.concatenate([base_emb, wrist_emb], axis=-1)
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")


def decode_dataset_names(
    tokenized_names: np.ndarray,
    tokenizer,
) -> list[str]:
    """Decode tokenized dataset names.
    
    Args:
        tokenized_names: Tokenized dataset names [batch, seq_len]
        tokenizer: Tokenizer to decode with
        
    Returns:
        List of decoded dataset name strings
    """
    dataset_names = []
    for i in range(tokenized_names.shape[0]):
        try:
            # Filter out padding tokens (usually 0)
            tokens = tokenized_names[i]
            # Decode and strip whitespace
            name = tokenizer.decode(tokens.tolist())
            name = name.strip()
            # Remove any special tokens or padding artifacts
            name = name.replace("<pad>", "").replace("<eos>", "").strip()
            if not name:
                name = "unknown"
            dataset_names.append(name)
        except Exception as e:
            logger.warning(f"Failed to decode dataset name for sample {i}: {e}")
            dataset_names.append("unknown")
    return dataset_names


def extract_all_embeddings(
    extractor: EmbeddingExtractor,
    data_loader,
    embedding_type: EmbeddingType,
    num_samples: int,
    data_sharding,
    mesh,
) -> EmbeddingResult:
    """Extract embeddings from all samples in the data loader.
    
    Args:
        extractor: Embedding extractor
        data_loader: Data loader
        embedding_type: Type of embedding to extract
        num_samples: Maximum number of samples to extract
        data_sharding: JAX sharding for data
        mesh: JAX mesh
        
    Returns:
        EmbeddingResult with all extracted embeddings
    """
    all_embeddings = []
    all_dataset_names = []
    sample_indices = []
    
    # Get tokenizer from data_loader for decoding dataset names
    tokenizer = data_loader.tokenizer
    
    # JIT compile the extraction function
    @jax.jit
    def extract_fn(obs):
        return extractor.extract(obs, embedding_type)
    
    data_iter = iter(data_loader)
    collected = 0
    
    pbar = tqdm.tqdm(
        total=num_samples,
        desc=f"Extracting {embedding_type} embeddings",
    )
    
    with sharding.set_mesh(mesh):
        while collected < num_samples:
            try:
                batch = next(data_iter)
            except StopIteration:
                logger.info("Reached end of dataset")
                break
            
            observation, _ = batch
            
            # Extract embeddings
            embeddings = extract_fn(observation)
            embeddings_np = np.array(jax.device_get(embeddings))
            batch_size = embeddings_np.shape[0]
            
            # Get dataset names from the batch
            dataset_names = ["unknown"] * batch_size
            
            # Try to get dataset names from tokenized_dataset_name in observation
            if hasattr(observation, 'tokenized_dataset_name') and observation.tokenized_dataset_name is not None:
                tokenized_names = np.array(training_utils.to_local_array(observation.tokenized_dataset_name))
                if tokenizer is not None:
                    dataset_names = decode_dataset_names(tokenized_names, tokenizer)
                else:
                    logger.warning("No tokenizer available, cannot decode dataset names")
            
            # Store results
            for i in range(batch_size):
                if collected >= num_samples:
                    break
                all_embeddings.append(embeddings_np[i])
                all_dataset_names.append(dataset_names[i] if i < len(dataset_names) else "unknown")
                sample_indices.append(collected)
                collected += 1
            
            pbar.update(min(batch_size, num_samples - collected + batch_size))
    
    pbar.close()
    
    return EmbeddingResult(
        embeddings=np.stack(all_embeddings),
        dataset_names=all_dataset_names,
        sample_indices=sample_indices,
    )


def reduce_dimensions(
    embeddings: np.ndarray,
    method: Literal["tsne", "umap", "pca"] = "tsne",
    n_components: int = 2,
    **kwargs,
) -> np.ndarray:
    """Reduce embedding dimensions for visualization.
    
    Args:
        embeddings: Input embeddings [num_samples, embedding_dim]
        method: Dimensionality reduction method
        n_components: Number of output dimensions
        **kwargs: Additional arguments for the reducer
        
    Returns:
        Reduced embeddings [num_samples, n_components]
    """
    logger.info(f"Reducing dimensions with {method} from {embeddings.shape[1]} to {n_components}")
    
    if method == "tsne":
        reducer = TSNE(
            n_components=n_components,
            perplexity=kwargs.get("perplexity", 30),
            random_state=kwargs.get("random_state", 42),
            n_iter=kwargs.get("n_iter", 1000),
        )
        return reducer.fit_transform(embeddings)
    
    elif method == "umap":
        if not HAS_UMAP:
            logger.warning("UMAP not installed, falling back to t-SNE")
            return reduce_dimensions(embeddings, "tsne", n_components, **kwargs)
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=kwargs.get("n_neighbors", 15),
            min_dist=kwargs.get("min_dist", 0.1),
            random_state=kwargs.get("random_state", 42),
        )
        return reducer.fit_transform(embeddings)
    
    elif method == "pca":
        reducer = PCA(n_components=n_components, random_state=kwargs.get("random_state", 42))
        return reducer.fit_transform(embeddings)
    
    else:
        raise ValueError(f"Unknown reduction method: {method}")


def compute_distance_metrics(
    result: EmbeddingResult,
    train_datasets: list[str] | None = None,
    eval_datasets: list[str] | None = None,
) -> dict[str, float]:
    """Compute distance metrics between embodiment clusters.
    
    Args:
        result: Embedding result with dataset labels
        train_datasets: List of training dataset names
        eval_datasets: List of evaluation dataset names
        
    Returns:
        Dictionary of distance metrics
    """
    # Group embeddings by dataset
    embeddings_by_dataset = defaultdict(list)
    for emb, name in zip(result.embeddings, result.dataset_names):
        embeddings_by_dataset[name].append(emb)
    
    # Convert to numpy arrays and compute centroids
    centroids = {}
    for name, embs in embeddings_by_dataset.items():
        embs_arr = np.stack(embs)
        centroids[name] = np.mean(embs_arr, axis=0)
    
    # Auto-detect train/eval splits if not provided
    all_datasets = list(centroids.keys())
    if train_datasets is None:
        train_datasets = all_datasets
    if eval_datasets is None:
        eval_datasets = all_datasets
    
    metrics = {}
    
    # Compute pairwise distances between all datasets
    for name1 in all_datasets:
        for name2 in all_datasets:
            if name1 != name2:
                dist = np.linalg.norm(centroids[name1] - centroids[name2])
                metrics[f"dist_{name1}_to_{name2}"] = float(dist)
    
    # Compute mean distance to nearest training dataset for each eval dataset
    for eval_name in eval_datasets:
        if eval_name not in centroids:
            continue
        
        distances_to_train = []
        for train_name in train_datasets:
            if train_name not in centroids or train_name == eval_name:
                continue
            dist = np.linalg.norm(centroids[eval_name] - centroids[train_name])
            distances_to_train.append(dist)
        
        if distances_to_train:
            metrics[f"{eval_name}_nearest_train_dist"] = float(min(distances_to_train))
            metrics[f"{eval_name}_mean_train_dist"] = float(np.mean(distances_to_train))
    
    # Compute intra-cluster variance for each dataset
    for name, embs in embeddings_by_dataset.items():
        if len(embs) > 1:
            embs_arr = np.stack(embs)
            variance = np.mean(np.var(embs_arr, axis=0))
            metrics[f"{name}_intra_variance"] = float(variance)
    
    return metrics


def plot_embeddings(
    embeddings_2d: np.ndarray,
    dataset_names: list[str],
    output_path: str,
    title: str = "Embodiment Embeddings",
    figsize: tuple[int, int] = (12, 10),
):
    """Create a scatter plot of embeddings colored by dataset.
    
    Args:
        embeddings_2d: 2D embeddings [num_samples, 2]
        dataset_names: Dataset name for each sample
        output_path: Path to save the figure
        title: Plot title
        figsize: Figure size
    """
    # Get unique datasets and assign colors
    unique_datasets = sorted(set(dataset_names))
    num_datasets = len(unique_datasets)
    
    # Use a colormap with enough distinct colors
    if num_datasets <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
    elif num_datasets <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
    else:
        colors = plt.cm.rainbow(np.linspace(0, 1, num_datasets))
    
    dataset_to_color = {name: colors[i] for i, name in enumerate(unique_datasets)}
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each dataset separately for legend
    for dataset in unique_datasets:
        mask = np.array([n == dataset for n in dataset_names])
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[dataset_to_color[dataset]],
            label=dataset,
            alpha=0.6,
            s=30,
        )
    
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved plot to {output_path}")


def plot_distance_heatmap(
    metrics: dict[str, float],
    output_path: str,
    title: str = "Cross-Embodiment Distances",
):
    """Create a heatmap of pairwise distances between embodiments.
    
    Args:
        metrics: Distance metrics dictionary
        output_path: Path to save the figure
        title: Plot title
    """
    # Extract pairwise distances
    dist_keys = [k for k in metrics.keys() if k.startswith("dist_")]
    if not dist_keys:
        logger.warning("No pairwise distances found in metrics")
        return
    
    # Parse dataset names from keys
    datasets = set()
    for key in dist_keys:
        parts = key.replace("dist_", "").split("_to_")
        if len(parts) == 2:
            datasets.add(parts[0])
            datasets.add(parts[1])
    
    datasets = sorted(datasets)
    n = len(datasets)
    
    # Build distance matrix
    dist_matrix = np.zeros((n, n))
    for i, d1 in enumerate(datasets):
        for j, d2 in enumerate(datasets):
            if i == j:
                dist_matrix[i, j] = 0
            else:
                key = f"dist_{d1}_to_{d2}"
                if key in metrics:
                    dist_matrix[i, j] = metrics[key]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(dist_matrix, cmap='viridis')
    
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(datasets, fontsize=8)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Distance", rotation=-90, va="bottom")
    
    # Add text annotations
    for i in range(n):
        for j in range(n):
            text = ax.text(j, i, f"{dist_matrix[i, j]:.2f}",
                          ha="center", va="center", color="w" if dist_matrix[i, j] > dist_matrix.max()/2 else "black",
                          fontsize=6)
    
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved heatmap to {output_path}")


def compare_models(
    model_configs: list[tuple[str, str, int | None]],  # (config_name, exp_name, step)
    embedding_type: EmbeddingType,
    num_samples: int,
    output_dir: str,
    split: str = "train",
    seed: int = 42,
    batch_size: int | None = None,
) -> dict[str, dict[str, float]]:
    """Compare embodiment embeddings across multiple models.
    
    Args:
        model_configs: List of (config_name, exp_name, checkpoint_step) tuples
        embedding_type: Type of embedding to extract
        num_samples: Number of samples per model
        output_dir: Output directory
        split: Data split to use
        seed: Random seed
        batch_size: Override batch size
        
    Returns:
        Dictionary mapping model names to their distance metrics
    """
    all_results = {}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for config_name, exp_name, step in model_configs:
        model_name = f"{config_name}_{exp_name}_step{step or 'latest'}"
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing model: {model_name}")
        logger.info(f"{'='*60}")
        
        try:
            # Get config
            config = _config.get_config(config_name)
            config = dataclasses.replace(config, exp_name=exp_name)
            if batch_size:
                config = dataclasses.replace(config, batch_size=batch_size)
            
            # Initialize runtime
            effective_fsdp = init_tpu(config)
            
            # Load checkpoint
            train_state, data_sharding, mesh = load_checkpoint(config, effective_fsdp, step)
            
            # Create data loader
            data_loader = _data_loader.create_data_loader(
                config,
                sharding=data_sharding,
                shuffle=True,
                split=split,
                seed=seed,
            )
            
            # Extract embeddings
            extractor = EmbeddingExtractor(train_state)
            result = extract_all_embeddings(
                extractor=extractor,
                data_loader=data_loader,
                embedding_type=embedding_type,
                num_samples=num_samples,
                data_sharding=data_sharding,
                mesh=mesh,
            )
            
            # Compute metrics
            metrics = compute_distance_metrics(result)
            all_results[model_name] = {
                "metrics": metrics,
                "embeddings": result.embeddings,
                "dataset_names": result.dataset_names,
            }
            
            # Save individual model results
            model_output_dir = output_path / model_name
            model_output_dir.mkdir(exist_ok=True)
            np.savez(
                model_output_dir / f"embeddings_{embedding_type}.npz",
                embeddings=result.embeddings,
                dataset_names=result.dataset_names,
            )
            
            with open(model_output_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"Saved results for {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to process {model_name}: {e}")
            continue
    
    # Create comparison plots
    if len(all_results) > 1:
        plot_model_comparison(all_results, output_path, embedding_type)
    
    return {name: res["metrics"] for name, res in all_results.items()}


def plot_model_comparison(
    all_results: dict[str, dict],
    output_dir: Path,
    embedding_type: str,
):
    """Create comparison plots across multiple models.
    
    Args:
        all_results: Dictionary mapping model names to their results
        output_dir: Output directory
        embedding_type: Embedding type used
    """
    # Collect all unique datasets across all models
    all_datasets = set()
    for res in all_results.values():
        all_datasets.update(set(res["dataset_names"]))
    all_datasets = sorted(all_datasets)
    
    # Create comparison bar chart for nearest-train distances
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Nearest train distance per dataset
    model_names = list(all_results.keys())
    x = np.arange(len(all_datasets))
    width = 0.8 / len(model_names)
    
    for i, (model_name, res) in enumerate(all_results.items()):
        metrics = res["metrics"]
        distances = []
        for dataset in all_datasets:
            key = f"{dataset}_nearest_train_dist"
            distances.append(metrics.get(key, 0))
        
        axes[0].bar(x + i * width, distances, width, label=model_name[:30])
    
    axes[0].set_xlabel("Dataset")
    axes[0].set_ylabel("Nearest Train Distance")
    axes[0].set_title("Cross-Embodiment Transfer: Distance to Nearest Training Embodiment")
    axes[0].set_xticks(x + width * (len(model_names) - 1) / 2)
    axes[0].set_xticklabels(all_datasets, rotation=45, ha='right', fontsize=8)
    axes[0].legend(fontsize=8)
    
    # Plot 2: Mean intra-cluster variance
    intra_variances = []
    for model_name, res in all_results.items():
        metrics = res["metrics"]
        variances = [v for k, v in metrics.items() if k.endswith("_intra_variance")]
        mean_var = np.mean(variances) if variances else 0
        intra_variances.append((model_name[:30], mean_var))
    
    names, values = zip(*intra_variances) if intra_variances else ([], [])
    axes[1].bar(names, values)
    axes[1].set_xlabel("Model")
    axes[1].set_ylabel("Mean Intra-Cluster Variance")
    axes[1].set_title("Embedding Cluster Tightness (lower = better separation)")
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"model_comparison_{embedding_type}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create side-by-side t-SNE plots
    n_models = len(all_results)
    if n_models > 0:
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
        if n_models == 1:
            axes = [axes]
        
        for ax, (model_name, res) in zip(axes, all_results.items()):
            embeddings_2d = reduce_dimensions(res["embeddings"], method="tsne")
            
            unique_datasets = sorted(set(res["dataset_names"]))
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_datasets)))
            dataset_to_color = {name: colors[i] for i, name in enumerate(unique_datasets)}
            
            for dataset in unique_datasets:
                mask = np.array([n == dataset for n in res["dataset_names"]])
                ax.scatter(
                    embeddings_2d[mask, 0],
                    embeddings_2d[mask, 1],
                    c=[dataset_to_color[dataset]],
                    label=dataset[:15],
                    alpha=0.6,
                    s=20,
                )
            
            ax.set_title(model_name[:40], fontsize=10)
            ax.legend(fontsize=6, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"tsne_comparison_{embedding_type}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Saved comparison plots to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize embodiment embeddings")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Single model extraction
    single_parser = subparsers.add_parser("single", help="Extract embeddings from a single model")
    single_parser.add_argument("--config", type=str, required=True, help="Config name")
    single_parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    single_parser.add_argument("--checkpoint_step", type=int, default=None, help="Checkpoint step")
    single_parser.add_argument("--num_samples", type=int, default=500, help="Number of samples")
    single_parser.add_argument("--embedding_type", type=str, default="prefix_prelogits",
                              choices=["image", "prefix_prelogits", "combined_prefix", "text_only", "per_camera", "wrist_image"])
    single_parser.add_argument("--reduction_method", type=str, default="tsne",
                              choices=["tsne", "umap", "pca"])
    single_parser.add_argument("--output_dir", type=str, default="./embeddings_viz")
    single_parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    single_parser.add_argument("--seed", type=int, default=42)
    single_parser.add_argument("--batch_size", type=int, default=None)
    
    # Multi-model comparison
    compare_parser = subparsers.add_parser("compare", help="Compare embeddings across multiple models")
    compare_parser.add_argument("--models", type=str, required=True,
                               help="Comma-separated list of config:exp_name:step (step optional)")
    compare_parser.add_argument("--num_samples", type=int, default=500)
    compare_parser.add_argument("--embedding_type", type=str, default="prefix_prelogits",
                               choices=["image", "prefix_prelogits", "combined_prefix", "text_only", "per_camera", "wrist_image"])
    compare_parser.add_argument("--output_dir", type=str, default="./embeddings_comparison")
    compare_parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    compare_parser.add_argument("--seed", type=int, default=42)
    compare_parser.add_argument("--batch_size", type=int, default=None)
    
    args = parser.parse_args()
    
    if args.command == "single" or args.command is None:
        # Default to single model if no subcommand
        if args.command is None:
            # Re-parse with single model arguments for backward compatibility
            parser = argparse.ArgumentParser(description="Visualize embodiment embeddings")
            parser.add_argument("--config", type=str, required=True, help="Config name")
            parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
            parser.add_argument("--checkpoint_step", type=int, default=None, help="Checkpoint step")
            parser.add_argument("--num_samples", type=int, default=500, help="Number of samples")
            parser.add_argument("--embedding_type", type=str, default="prefix_prelogits",
                               choices=["image", "prefix_prelogits", "combined_prefix", "text_only", "per_camera", "wrist_image"])
            parser.add_argument("--reduction_method", type=str, default="tsne",
                               choices=["tsne", "umap", "pca"])
            parser.add_argument("--output_dir", type=str, default="./embeddings_viz")
            parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
            parser.add_argument("--seed", type=int, default=42)
            parser.add_argument("--batch_size", type=int, default=None)
            args = parser.parse_args()
        
        run_single_model(args)
    
    elif args.command == "compare":
        # Parse model specifications
        model_configs = []
        for model_spec in args.models.split(","):
            parts = model_spec.strip().split(":")
            if len(parts) == 2:
                model_configs.append((parts[0], parts[1], None))
            elif len(parts) == 3:
                model_configs.append((parts[0], parts[1], int(parts[2]) if parts[2] else None))
            else:
                logger.error(f"Invalid model specification: {model_spec}")
                continue
        
        if not model_configs:
            logger.error("No valid model configurations provided")
            return
        
        compare_models(
            model_configs=model_configs,
            embedding_type=args.embedding_type,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            split=args.split,
            seed=args.seed,
            batch_size=args.batch_size,
        )


def run_single_model(args):
    """Run embedding extraction for a single model."""
    # Get config
    config = _config.get_config(args.config)
    config = dataclasses.replace(config, exp_name=args.exp_name)
    if args.batch_size:
        config = dataclasses.replace(config, batch_size=args.batch_size)
    
    logger.info(f"Config: {config.name}")
    logger.info(f"Checkpoint dir: {config.checkpoint_dir}")
    
    # Initialize runtime
    effective_fsdp = init_tpu(config)
    
    # Load checkpoint
    train_state, data_sharding, mesh = load_checkpoint(
        config, effective_fsdp, args.checkpoint_step
    )
    
    # Create data loader
    logger.info("Creating data loader...")
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
        split=args.split,
        seed=args.seed,
    )
    
    # Create embedding extractor
    extractor = EmbeddingExtractor(train_state)
    
    # Extract embeddings
    logger.info(f"Extracting {args.embedding_type} embeddings...")
    result = extract_all_embeddings(
        extractor=extractor,
        data_loader=data_loader,
        embedding_type=args.embedding_type,
        num_samples=args.num_samples,
        data_sharding=data_sharding,
        mesh=mesh,
    )
    
    logger.info(f"Extracted {len(result.embeddings)} embeddings")
    logger.info(f"Unique datasets: {sorted(set(result.dataset_names))}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw embeddings
    np.savez(
        output_dir / f"embeddings_{args.embedding_type}.npz",
        embeddings=result.embeddings,
        dataset_names=result.dataset_names,
        sample_indices=result.sample_indices,
    )
    logger.info(f"Saved raw embeddings to {output_dir / f'embeddings_{args.embedding_type}.npz'}")
    
    # Reduce dimensions
    embeddings_2d = reduce_dimensions(result.embeddings, method=args.reduction_method)
    
    # Compute metrics
    metrics = compute_distance_metrics(result)
    
    # Save metrics
    metrics_path = output_dir / f"metrics_{args.embedding_type}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")
    
    # Print key metrics
    logger.info("=" * 60)
    logger.info("KEY METRICS")
    logger.info("=" * 60)
    for key, value in sorted(metrics.items()):
        logger.info(f"{key}: {value:.4f}")
    
    # Create plots
    plot_embeddings(
        embeddings_2d=embeddings_2d,
        dataset_names=result.dataset_names,
        output_path=str(output_dir / f"scatter_{args.embedding_type}_{args.reduction_method}.png"),
        title=f"Embodiment Embeddings ({args.embedding_type}, {args.reduction_method})",
    )
    
    plot_distance_heatmap(
        metrics=metrics,
        output_path=str(output_dir / f"heatmap_{args.embedding_type}.png"),
        title=f"Cross-Embodiment Distances ({args.embedding_type})",
    )
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
