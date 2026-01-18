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
    

def init_tpu(config: _config.TrainConfig, fsdp_devices_override: int | None = None):
    """Initialize TPU/GPU runtime.
    
    Note: JAX distributed should be initialized before this function is called.
    See initialize_distributed() which should be called at program start.
    
    Args:
        config: Training config
        fsdp_devices_override: If provided, override config.fsdp_devices
        
    Returns:
        Effective FSDP devices to use
    """
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
    
    # Use override if provided, otherwise use config value
    fsdp_devices = fsdp_devices_override if fsdp_devices_override is not None else config.fsdp_devices
    
    # Determine effective FSDP devices
    process_count = jax.process_count()
    local_devices = jax.local_device_count()
    global_devices = jax.device_count()
    logger.info(f"Devices: local={local_devices}, global={global_devices}, processes={process_count}")
    logger.info(f"FSDP devices: config={config.fsdp_devices}, override={fsdp_devices_override}, using={fsdp_devices}")
    
    if process_count == 1:
        target = min(fsdp_devices, local_devices)
        effective_fsdp = 1
        for d in range(target, 0, -1):
            if global_devices % d == 0:
                effective_fsdp = d
                break
    else:
        effective_fsdp = fsdp_devices
    
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
    
    # Create optimizer and training state shape
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
    
    # Try standard checkpoint loading first, fall back to simple PyTree loading
    try:
        # Initialize checkpoint manager (may fail if distributed not initialized)
        checkpoint_manager, _ = _checkpoints.initialize_checkpoint_dir(
            config.checkpoint_dir,
            keep_period=config.keep_period,
            overwrite=False,
            resume=True,
            async_enable=False,
        )
        
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
    except ValueError as e:
        if "Distributed system is not available" in str(e):
            logger.warning("Distributed system not available, using simple checkpoint loading")
            train_state = _load_checkpoint_simple(
                config, train_state_shape, train_state_sharding, checkpoint_step
            )
        else:
            raise
    
    # Use EMA params if available
    if train_state.ema_params is not None:
        logger.info("Using EMA params")
        train_state = dataclasses.replace(train_state, params=train_state.ema_params)
    
    logger.info(f"Loaded checkpoint at step {train_state.step}")
    
    return train_state, data_sharding, mesh


def _load_checkpoint_simple(
    config: _config.TrainConfig,
    train_state_shape: training_utils.TrainState,
    train_state_sharding,
    checkpoint_step: int | None = None,
) -> training_utils.TrainState:
    """Simple checkpoint loading without CheckpointManager (no distributed required)."""
    import orbax.checkpoint as ocp
    
    checkpoint_dir = epath.Path(config.checkpoint_dir)
    
    # Find available checkpoints by listing directories
    if str(checkpoint_dir).startswith("gs://"):
        # GCS path - list using gcsfs or glob
        import subprocess
        result = subprocess.run(
            ["gsutil", "ls", str(checkpoint_dir)],
            capture_output=True, text=True
        )
        subdirs = [p.rstrip('/').split('/')[-1] for p in result.stdout.strip().split('\n') if p]
    else:
        # Local path
        subdirs = [p.name for p in checkpoint_dir.iterdir() if p.is_dir()]
    
    # Filter to numeric step directories
    available_steps = sorted([int(d) for d in subdirs if d.isdigit()])
    logger.info(f"Available checkpoints (simple): {available_steps}")
    
    if checkpoint_step is None:
        checkpoint_step = max(available_steps) if available_steps else None
        logger.info(f"Using latest checkpoint: {checkpoint_step}")
    
    if checkpoint_step is None or checkpoint_step not in available_steps:
        raise ValueError(f"Step {checkpoint_step} not found. Available: {available_steps}")
    
    step_dir = checkpoint_dir / str(checkpoint_step)
    
    # Try to load train_state or params
    checkpointer = ocp.PyTreeCheckpointer()
    
    # Check what items are available
    train_state_dir = step_dir / "train_state"
    params_dir = step_dir / "params"
    
    if train_state_dir.exists():
        logger.info(f"Loading train_state from {train_state_dir}")
        restore_args = ocp.checkpoint_utils.construct_restore_args(train_state_shape)
        train_state = checkpointer.restore(
            train_state_dir,
            item=train_state_shape,
            restore_args=restore_args,
        )
    elif params_dir.exists():
        logger.info(f"Loading params from {params_dir}")
        # Load just params and construct train_state
        restore_args = ocp.checkpoint_utils.construct_restore_args(train_state_shape.params)
        params = checkpointer.restore(
            params_dir,
            item=train_state_shape.params,
            restore_args=restore_args,
        )
        train_state = dataclasses.replace(train_state_shape, params=params, step=checkpoint_step)
    else:
        raise ValueError(f"No train_state or params found in {step_dir}")
    
    return train_state


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
        
        # Check if model uses action training (requires 2-element lists)
        enable_action_training = getattr(self.model, 'enable_action_training', False)
        
        # Forward through LLM - list length must match model's internal config
        if enable_action_training:
            # Model expects [prefix, suffix] format
            pre_logits, _ = self.model.PaliGemma.llm(
                [prefix_tokens, None],
                mask=prefix_attn_mask,
                positions=positions,
                adarms_cond=[None, None],
            )
        else:
            # Model expects [prefix] format
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
    
    # Create output sharding (replicated so we can easily fetch to CPU)
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    
    # Create a JIT-compiled extraction function with proper output sharding
    def _extract_impl(observation):
        return extractor.extract(observation, embedding_type)
    
    # We'll compile after seeing the first batch to get proper input shardings
    extract_fn = None
    
    data_iter = iter(data_loader)
    collected = 0
    first_batch = True
    
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
            
            # Get local batch for dataset names (before extraction)
            tokenized_names = None
            if hasattr(observation, 'tokenized_dataset_name') and observation.tokenized_dataset_name is not None:
                tokenized_names = training_utils.to_local_array(observation.tokenized_dataset_name)
            
            # Create JIT function on first batch with proper input sharding structure
            if extract_fn is None:
                # Build input sharding that matches observation structure
                def get_sharding(x):
                    if hasattr(x, 'shape') and len(x.shape) > 0:
                        return data_sharding
                    return replicated_sharding
                
                in_shardings = jax.tree_util.tree_map(get_sharding, observation)
                
                extract_fn = jax.jit(
                    _extract_impl,
                    in_shardings=in_shardings,
                    out_shardings=replicated_sharding,
                )
                logger.info("JIT-compiled extraction function")
            
            # Extract embeddings with JIT-compiled function
            try:
                embeddings = extract_fn(observation)
            except Exception as e:
                if first_batch:
                    logger.warning(f"JIT extraction failed, falling back to direct extraction: {e}")
                # Fallback: run without JIT
                embeddings = _extract_impl(observation)
            
            first_batch = False
            
            # Convert to numpy - replicated sharding should make this straightforward
            embeddings_np = np.asarray(embeddings)
            batch_size = embeddings_np.shape[0]
            
            # Get dataset names from the batch
            dataset_names = ["unknown"] * batch_size
            
            if tokenized_names is not None and tokenizer is not None:
                tokenized_names_np = np.asarray(tokenized_names)
                dataset_names = decode_dataset_names(tokenized_names_np, tokenizer)
            
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
        # Uspe max_iter for scikit-learn >= 1.2, fallback to n_iter for older versions
        tsne_kwargs = {
            "n_components": n_components,
            "perplexity": kwargs.get("perplexity", 30),
            "random_state": kwargs.get("random_state", 42),
        }
        # Try max_iter first (scikit-learn >= 1.2), then n_iter for older versions
        try:
            reducer = TSNE(**tsne_kwargs, max_iter=kwargs.get("max_iter", 1000))
        except TypeError:
            reducer = TSNE(**tsne_kwargs, n_iter=kwargs.get("n_iter", 1000))
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
    fsdp_devices: int | None = None,
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
        fsdp_devices: Override FSDP devices (default: use config value)
        
    Returns:
        Dictionary mapping model names to their distance metrics
    """
    all_results = {}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize runtime once using the first model's config
    first_config_name, first_exp_name, _ = model_configs[0]
    first_config = _config.get_config(first_config_name)
    first_config = dataclasses.replace(first_config, exp_name=first_exp_name)
    effective_fsdp = init_tpu(first_config, fsdp_devices_override=fsdp_devices)
    
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
            
            # Load checkpoint (using pre-computed effective_fsdp)
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


def compare_cross_embodiment(
    model_configs: list[tuple[str, str, int | None]],  # (config_name, exp_name, step)
    target_dataset: str,
    embedding_type: EmbeddingType,
    num_samples: int,
    target_num_samples: int,
    reduction_method: str,
    output_dir: str,
    split: str = "train",
    seed: int = 42,
    batch_size: int | None = None,
    separate_tsne: bool = False,
    fsdp_devices: int | None = None,
) -> dict[str, dict[str, float]]:
    """Compare cross-embodiment transfer across multiple models with shared t-SNE.
    
    This function:
    1. Extracts embeddings from each model for both training mix AND target dataset
    2. Combines ALL embeddings together
    3. Runs t-SNE once on the combined set (so distances are comparable)
    4. Optionally runs t-SNE separately for each model and plots side-by-side
    5. Creates visualizations comparing cross-embodiment transfer
    
    Args:
        model_configs: List of (config_name, exp_name, checkpoint_step) tuples
        target_dataset: Target dataset name to compare against
        embedding_type: Type of embedding to extract
        num_samples: Number of samples from training mix per model
        target_num_samples: Number of samples from target dataset per model
        reduction_method: Dimensionality reduction method (tsne/umap/pca)
        output_dir: Output directory
        split: Data split to use
        seed: Random seed
        batch_size: Override batch size
        separate_tsne: If True, also run t-SNE separately per model and plot side-by-side
        fsdp_devices: Override FSDP devices (default: use config value)
        
    Returns:
        Dictionary mapping model names to their distance metrics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Structure to hold all embeddings
    # Key: model_name, Value: {"train": embeddings, "target": embeddings, "train_names": [...], "target_names": [...]}
    model_embeddings = {}
    
    # Initialize runtime once using the first model's config
    first_config_name, first_exp_name, _ = model_configs[0]
    first_config = _config.get_config(first_config_name)
    first_config = dataclasses.replace(first_config, exp_name=first_exp_name)
    effective_fsdp = init_tpu(first_config, fsdp_devices_override=fsdp_devices)
    
    for config_name, exp_name, step in model_configs:
        model_name = f"{exp_name}_step{step or 'latest'}"
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing model: {model_name}")
        logger.info(f"{'='*60}")
        
        try:
            # Get config
            config = _config.get_config(config_name)
            config = dataclasses.replace(config, exp_name=exp_name)
            if batch_size:
                config = dataclasses.replace(config, batch_size=batch_size)
            
            # Load checkpoint (using pre-computed effective_fsdp)
            train_state, data_sharding, mesh = load_checkpoint(config, effective_fsdp, step)
            
            # Create embedding extractor
            extractor = EmbeddingExtractor(train_state)
            
            # --- Extract training mix embeddings ---
            logger.info(f"Creating training data loader...")
            train_data_loader = _data_loader.create_data_loader(
                config,
                sharding=data_sharding,
                shuffle=True,
                split=split,
                seed=seed,
            )
            
            logger.info(f"Extracting {embedding_type} embeddings from training mix...")
            train_result = extract_all_embeddings(
                extractor=extractor,
                data_loader=train_data_loader,
                embedding_type=embedding_type,
                num_samples=num_samples,
                data_sharding=data_sharding,
                mesh=mesh,
            )
            logger.info(f"Extracted {len(train_result.embeddings)} training embeddings")
            
            # --- Extract target dataset embeddings ---
            logger.info(f"Creating target data loader for {target_dataset}...")
            target_config = create_target_config(config, target_dataset)
            target_data_loader = _data_loader.create_data_loader(
                target_config,
                sharding=data_sharding,
                shuffle=True,
                split=split,
                seed=seed + 1,
            )
            
            logger.info(f"Extracting {embedding_type} embeddings from target dataset...")
            target_result = extract_all_embeddings(
                extractor=extractor,
                data_loader=target_data_loader,
                embedding_type=embedding_type,
                num_samples=target_num_samples,
                data_sharding=data_sharding,
                mesh=mesh,
            )
            logger.info(f"Extracted {len(target_result.embeddings)} target embeddings")
            
            # Store results
            model_embeddings[model_name] = {
                "train_embeddings": train_result.embeddings,
                "train_names": train_result.dataset_names,
                "target_embeddings": target_result.embeddings,
                "target_names": target_result.dataset_names,
            }
            
        except Exception as e:
            logger.error(f"Failed to process {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(model_embeddings) == 0:
        logger.error("No models processed successfully")
        return {}
    
    # --- Combine all embeddings for shared t-SNE ---
    logger.info(f"\n{'='*60}")
    logger.info("Combining all embeddings for shared dimensionality reduction...")
    logger.info(f"{'='*60}")
    
    all_embeddings = []
    all_labels = []  # (model_name, "train"/"target")
    all_dataset_names = []
    
    for model_name, data in model_embeddings.items():
        # Add training embeddings
        for i, emb in enumerate(data["train_embeddings"]):
            all_embeddings.append(emb)
            all_labels.append((model_name, "train"))
            all_dataset_names.append(data["train_names"][i] if i < len(data["train_names"]) else "unknown")
        
        # Add target embeddings
        for i, emb in enumerate(data["target_embeddings"]):
            all_embeddings.append(emb)
            all_labels.append((model_name, "target"))
            all_dataset_names.append(data["target_names"][i] if i < len(data["target_names"]) else target_dataset)
    
    all_embeddings = np.stack(all_embeddings)
    logger.info(f"Total embeddings: {len(all_embeddings)}")
    
    # --- Run dimensionality reduction once on combined data ---
    logger.info(f"Running {reduction_method} on combined embeddings...")
    embeddings_2d = reduce_dimensions(all_embeddings, method=reduction_method)
    
    # --- Compute metrics for each model ---
    metrics_per_model = {}
    
    for model_name in model_embeddings.keys():
        # Get indices for this model
        train_mask = np.array([l[0] == model_name and l[1] == "train" for l in all_labels])
        target_mask = np.array([l[0] == model_name and l[1] == "target" for l in all_labels])
        
        if np.any(train_mask) and np.any(target_mask):
            # Compute centroids in original high-dim space
            train_centroid_hd = np.mean(all_embeddings[train_mask], axis=0)
            target_centroid_hd = np.mean(all_embeddings[target_mask], axis=0)
            
            # Compute centroids in 2D space (for visualization)
            train_centroid_2d = np.mean(embeddings_2d[train_mask], axis=0)
            target_centroid_2d = np.mean(embeddings_2d[target_mask], axis=0)
            
            # Distance metrics
            dist_hd = float(np.linalg.norm(target_centroid_hd - train_centroid_hd))
            dist_2d = float(np.linalg.norm(target_centroid_2d - train_centroid_2d))
            
            metrics_per_model[model_name] = {
                "target_to_train_dist_hd": dist_hd,
                "target_to_train_dist_2d": dist_2d,
                "train_centroid_2d": train_centroid_2d.tolist(),
                "target_centroid_2d": target_centroid_2d.tolist(),
                "num_train_samples": int(np.sum(train_mask)),
                "num_target_samples": int(np.sum(target_mask)),
            }
    
    # --- Save metrics ---
    metrics_path = output_path / f"cross_embodiment_metrics_{target_dataset}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_per_model, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")
    
    # --- Print comparison ---
    logger.info(f"\n{'='*60}")
    logger.info(f"CROSS-EMBODIMENT TRANSFER COMPARISON (Target: {target_dataset})")
    logger.info(f"{'='*60}")
    logger.info(f"{'Model':<40} {'Train→Target Dist (HD)':<25} {'Train→Target Dist (2D)':<25}")
    logger.info("-" * 90)
    
    # Sort by distance (lower is better)
    sorted_models = sorted(metrics_per_model.items(), key=lambda x: x[1]["target_to_train_dist_hd"])
    for model_name, metrics in sorted_models:
        logger.info(
            f"{model_name:<40} {metrics['target_to_train_dist_hd']:<25.4f} {metrics['target_to_train_dist_2d']:<25.4f}"
        )
    
    if len(sorted_models) >= 2:
        best_model = sorted_models[0][0]
        worst_model = sorted_models[-1][0]
        improvement = (
            (sorted_models[-1][1]["target_to_train_dist_hd"] - sorted_models[0][1]["target_to_train_dist_hd"])
            / sorted_models[-1][1]["target_to_train_dist_hd"]
            * 100
        )
        logger.info(f"\nBest model: {best_model}")
        logger.info(f"Worst model: {worst_model}")
        logger.info(f"Improvement: {improvement:.1f}% closer to target")
    
    # --- Create visualizations ---
    plot_cross_embodiment_comparison(
        embeddings_2d=embeddings_2d,
        all_labels=all_labels,
        model_names=list(model_embeddings.keys()),
        metrics_per_model=metrics_per_model,
        target_dataset=target_dataset,
        output_path=output_path,
        embedding_type=embedding_type,
        reduction_method=reduction_method,
    )
    
    # --- Separate t-SNE per model (side-by-side comparison) ---
    if separate_tsne:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running separate {reduction_method} for each model...")
        logger.info(f"{'='*60}")
        
        # Run t-SNE separately for each model
        separate_embeddings_2d = {}
        separate_metrics = {}
        
        for model_name, data in model_embeddings.items():
            # Combine this model's train + target embeddings
            model_all_emb = np.vstack([data["train_embeddings"], data["target_embeddings"]])
            model_labels = (
                ["train"] * len(data["train_embeddings"]) + 
                ["target"] * len(data["target_embeddings"])
            )
            
            logger.info(f"Running {reduction_method} for {model_name} ({len(model_all_emb)} samples)...")
            emb_2d = reduce_dimensions(model_all_emb, method=reduction_method)
            
            # Compute centroids in this model's 2D space
            train_mask = np.array([l == "train" for l in model_labels])
            target_mask = np.array([l == "target" for l in model_labels])
            
            train_centroid_2d = np.mean(emb_2d[train_mask], axis=0)
            target_centroid_2d = np.mean(emb_2d[target_mask], axis=0)
            dist_2d = float(np.linalg.norm(target_centroid_2d - train_centroid_2d))
            
            separate_embeddings_2d[model_name] = {
                "embeddings_2d": emb_2d,
                "labels": model_labels,
                "train_names": data["train_names"],
                "target_names": data["target_names"],
            }
            
            separate_metrics[model_name] = {
                "train_centroid_2d": train_centroid_2d.tolist(),
                "target_centroid_2d": target_centroid_2d.tolist(),
                "target_to_train_dist_2d": dist_2d,
            }
            
            logger.info(f"  {model_name}: dist_2d = {dist_2d:.2f}")
        
        # Plot side-by-side
        plot_separate_tsne_side_by_side(
            separate_embeddings_2d=separate_embeddings_2d,
            separate_metrics=separate_metrics,
            target_dataset=target_dataset,
            output_path=output_path,
            embedding_type=embedding_type,
            reduction_method=reduction_method,
        )
        
        # Save separate t-SNE metrics
        with open(output_path / f"separate_tsne_metrics_{target_dataset}.json", "w") as f:
            json.dump(separate_metrics, f, indent=2)
    
    # Save raw data
    np.savez(
        output_path / f"embeddings_comparison_{target_dataset}.npz",
        embeddings_2d=embeddings_2d,
        embeddings_hd=all_embeddings,
        labels=np.array(all_labels, dtype=object),
        dataset_names=np.array(all_dataset_names, dtype=object),
    )
    
    logger.info(f"\nDone! Results saved to {output_path}")
    
    return metrics_per_model


def plot_cross_embodiment_comparison(
    embeddings_2d: np.ndarray,
    all_labels: list[tuple[str, str]],
    model_names: list[str],
    metrics_per_model: dict[str, dict],
    target_dataset: str,
    output_path: Path,
    embedding_type: str,
    reduction_method: str,
):
    """Create visualizations for cross-embodiment comparison.
    
    Args:
        embeddings_2d: 2D embeddings [num_samples, 2]
        all_labels: List of (model_name, "train"/"target") for each sample
        model_names: List of model names
        metrics_per_model: Metrics dictionary per model
        target_dataset: Target dataset name
        output_path: Output directory
        embedding_type: Embedding type used
        reduction_method: Reduction method used
    """
    n_models = len(model_names)
    
    # Color palette for models
    if n_models <= 10:
        model_colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_models]
    else:
        model_colors = plt.cm.rainbow(np.linspace(0, 1, n_models))
    
    model_to_color = {name: model_colors[i] for i, name in enumerate(model_names)}
    
    # --- Plot 1: All embeddings with model colors, train=circle, target=star ---
    fig, ax = plt.subplots(figsize=(14, 10))
    
    for model_name in model_names:
        color = model_to_color[model_name]
        
        # Training points (circles)
        train_mask = np.array([l[0] == model_name and l[1] == "train" for l in all_labels])
        ax.scatter(
            embeddings_2d[train_mask, 0],
            embeddings_2d[train_mask, 1],
            c=[color],
            marker='o',
            alpha=0.4,
            s=30,
            label=f"{model_name[:25]} (train)",
        )
        
        # Target points (stars)
        target_mask = np.array([l[0] == model_name and l[1] == "target" for l in all_labels])
        ax.scatter(
            embeddings_2d[target_mask, 0],
            embeddings_2d[target_mask, 1],
            c=[color],
            marker='*',
            alpha=0.8,
            s=100,
            label=f"{model_name[:25]} (target)",
            edgecolors='black',
            linewidths=0.5,
        )
        
        # Plot centroids and connecting line
        if model_name in metrics_per_model:
            train_centroid = metrics_per_model[model_name]["train_centroid_2d"]
            target_centroid = metrics_per_model[model_name]["target_centroid_2d"]
            
            # Train centroid (square)
            ax.scatter(
                train_centroid[0], train_centroid[1],
                c=[color], marker='s', s=200, edgecolors='black', linewidths=2,
            )
            
            # Target centroid (pentagon)
            ax.scatter(
                target_centroid[0], target_centroid[1],
                c=[color], marker='p', s=200, edgecolors='black', linewidths=2,
            )
            
            # Line connecting centroids
            ax.plot(
                [train_centroid[0], target_centroid[0]],
                [train_centroid[1], target_centroid[1]],
                color=color, linestyle='--', linewidth=2, alpha=0.7,
            )
            
            # Distance annotation
            mid_point = [(train_centroid[0] + target_centroid[0]) / 2,
                        (train_centroid[1] + target_centroid[1]) / 2]
            dist = metrics_per_model[model_name]["target_to_train_dist_2d"]
            ax.annotate(
                f'd={dist:.1f}',
                xy=mid_point,
                fontsize=9,
                fontweight='bold',
                ha='center',
                va='bottom',
                color=color,
            )
    
    ax.set_xlabel("Dimension 1", fontsize=12)
    ax.set_ylabel("Dimension 2", fontsize=12)
    ax.set_title(
        f"Cross-Embodiment Transfer Comparison\n"
        f"Target: {target_dataset} | Embedding: {embedding_type} | Method: {reduction_method}\n"
        f"(circles=train, stars=target, squares=train centroid, pentagons=target centroid)",
        fontsize=11,
    )
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path / f"scatter_all_{embedding_type}_{reduction_method}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # --- Plot 2: Bar chart comparing distances ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sorted_models = sorted(metrics_per_model.items(), key=lambda x: x[1]["target_to_train_dist_hd"])
    names = [m[0][:30] for m in sorted_models]
    distances_hd = [m[1]["target_to_train_dist_hd"] for m in sorted_models]
    colors = [model_to_color[m[0]] for m in sorted_models]
    
    bars = ax.bar(names, distances_hd, color=colors, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, dist in zip(bars, distances_hd):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{dist:.1f}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold',
        )
    
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Distance to Target (High-Dim)", fontsize=12)
    ax.set_title(
        f"Cross-Embodiment Transfer: Distance from Training Mix to {target_dataset}\n"
        f"(Lower = Better Transfer)",
        fontsize=11,
    )
    ax.tick_params(axis='x', rotation=45)
    
    # Add ranking annotation
    if len(sorted_models) >= 1:
        ax.annotate(
            "← Better Transfer",
            xy=(0.02, 0.95),
            xycoords='axes fraction',
            fontsize=10,
            color='green',
            fontweight='bold',
        )
    
    plt.tight_layout()
    plt.savefig(output_path / f"bar_distances_{embedding_type}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # --- Plot 3: Separate subplots for each model ---
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    for ax, model_name in zip(axes, model_names):
        color = model_to_color[model_name]
        
        # All other models' points in very light gray
        other_mask = np.array([l[0] != model_name for l in all_labels])
        ax.scatter(
            embeddings_2d[other_mask, 0],
            embeddings_2d[other_mask, 1],
            c='lightgray',
            alpha=0.2,
            s=10,
        )
        
        # This model's training points
        train_mask = np.array([l[0] == model_name and l[1] == "train" for l in all_labels])
        ax.scatter(
            embeddings_2d[train_mask, 0],
            embeddings_2d[train_mask, 1],
            c=[color],
            marker='o',
            alpha=0.5,
            s=40,
            label='Training Mix',
        )
        
        # This model's target points
        target_mask = np.array([l[0] == model_name and l[1] == "target" for l in all_labels])
        ax.scatter(
            embeddings_2d[target_mask, 0],
            embeddings_2d[target_mask, 1],
            c='red',
            marker='*',
            alpha=0.8,
            s=100,
            label=f'Target ({target_dataset})',
            edgecolors='darkred',
            linewidths=0.5,
        )
        
        # Centroids
        if model_name in metrics_per_model:
            train_centroid = metrics_per_model[model_name]["train_centroid_2d"]
            target_centroid = metrics_per_model[model_name]["target_centroid_2d"]
            
            ax.scatter(train_centroid[0], train_centroid[1], c=[color], marker='X', s=200, edgecolors='black', linewidths=2)
            ax.scatter(target_centroid[0], target_centroid[1], c='red', marker='X', s=200, edgecolors='darkred', linewidths=2)
            
            ax.plot(
                [train_centroid[0], target_centroid[0]],
                [train_centroid[1], target_centroid[1]],
                'k--', linewidth=2, alpha=0.7,
            )
            
            dist = metrics_per_model[model_name]["target_to_train_dist_hd"]
            ax.set_title(f"{model_name[:35]}\nDist: {dist:.2f}", fontsize=10)
        else:
            ax.set_title(model_name[:35], fontsize=10)
        
        ax.legend(fontsize=8, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path / f"subplots_{embedding_type}_{reduction_method}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved visualization plots to {output_path}")


def plot_separate_tsne_side_by_side(
    separate_embeddings_2d: dict[str, dict],
    separate_metrics: dict[str, dict],
    target_dataset: str,
    output_path: Path,
    embedding_type: str,
    reduction_method: str,
):
    """Plot separate t-SNE for each model side-by-side.
    
    Each model gets its own t-SNE, allowing comparison of the *shape* and *structure*
    of embeddings rather than absolute positions.
    
    Args:
        separate_embeddings_2d: Dict mapping model_name to {"embeddings_2d", "labels", ...}
        separate_metrics: Dict mapping model_name to metrics including centroids and distances
        target_dataset: Target dataset name
        output_path: Output directory
        embedding_type: Embedding type used
        reduction_method: Reduction method used
    """
    model_names = list(separate_embeddings_2d.keys())
    n_models = len(model_names)
    
    if n_models == 0:
        logger.warning("No models to plot for separate t-SNE")
        return
    
    # --- Main side-by-side plot ---
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6))
    if n_models == 1:
        axes = [axes]
    
    for ax, model_name in zip(axes, model_names):
        data = separate_embeddings_2d[model_name]
        emb_2d = data["embeddings_2d"]
        labels = data["labels"]
        
        # Masks
        train_mask = np.array([l == "train" for l in labels])
        target_mask = np.array([l == "target" for l in labels])
        
        # Plot training points (blue)
        ax.scatter(
            emb_2d[train_mask, 0],
            emb_2d[train_mask, 1],
            c='steelblue',
            marker='o',
            alpha=0.5,
            s=40,
            label='Training Mix',
        )
        
        # Plot target points (red stars)
        ax.scatter(
            emb_2d[target_mask, 0],
            emb_2d[target_mask, 1],
            c='crimson',
            marker='*',
            alpha=0.8,
            s=120,
            label=f'Target ({target_dataset})',
            edgecolors='darkred',
            linewidths=0.5,
        )
        
        # Plot centroids and line
        metrics = separate_metrics[model_name]
        train_centroid = metrics["train_centroid_2d"]
        target_centroid = metrics["target_centroid_2d"]
        dist = metrics["target_to_train_dist_2d"]
        
        # Train centroid (large blue X)
        ax.scatter(
            train_centroid[0], train_centroid[1],
            c='steelblue', marker='X', s=300,
            edgecolors='black', linewidths=2,
            label='Train Centroid', zorder=10,
        )
        
        # Target centroid (large red X)
        ax.scatter(
            target_centroid[0], target_centroid[1],
            c='crimson', marker='X', s=300,
            edgecolors='darkred', linewidths=2,
            label='Target Centroid', zorder=10,
        )
        
        # Connecting line
        ax.plot(
            [train_centroid[0], target_centroid[0]],
            [train_centroid[1], target_centroid[1]],
            'k--', linewidth=2.5, alpha=0.7, zorder=5,
        )
        
        # Distance annotation
        mid_point = [
            (train_centroid[0] + target_centroid[0]) / 2,
            (train_centroid[1] + target_centroid[1]) / 2
        ]
        ax.annotate(
            f'd = {dist:.1f}',
            xy=mid_point,
            fontsize=12,
            fontweight='bold',
            ha='center',
            va='bottom',
            color='black',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
        )
        
        # Title with distance
        ax.set_title(
            f"{model_name[:40]}\n(2D Distance: {dist:.2f})",
            fontsize=11,
            fontweight='bold',
        )
        ax.set_xlabel("Dimension 1", fontsize=10)
        ax.set_ylabel("Dimension 2", fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(
        f"Separate {reduction_method.upper()} per Model: Train vs Target ({target_dataset})\n"
        f"Embedding: {embedding_type}",
        fontsize=13,
        fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig(
        output_path / f"separate_tsne_side_by_side_{embedding_type}_{reduction_method}.png",
        dpi=150, bbox_inches='tight'
    )
    plt.close()
    
    # --- Comparison bar chart for separate t-SNE distances ---
    fig, ax = plt.subplots(figsize=(max(8, 3 * n_models), 5))
    
    sorted_models = sorted(separate_metrics.items(), key=lambda x: x[1]["target_to_train_dist_2d"])
    names = [m[0][:35] for m in sorted_models]
    distances = [m[1]["target_to_train_dist_2d"] for m in sorted_models]
    
    # Color gradient: green (best) to red (worst)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(distances)))
    
    bars = ax.bar(names, distances, color=colors, edgecolor='black', linewidth=1.5)
    
    # Value labels
    for bar, dist in zip(bars, distances):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f'{dist:.1f}',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold',
        )
    
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel(f"Train→Target Distance (Separate {reduction_method.upper()})", fontsize=12)
    ax.set_title(
        f"Cross-Embodiment Transfer Comparison (Separate {reduction_method.upper()})\n"
        f"Target: {target_dataset} | Lower = Better Transfer",
        fontsize=12,
        fontweight='bold',
    )
    ax.tick_params(axis='x', rotation=30)
    
    # Ranking annotation
    if len(sorted_models) >= 2:
        best = sorted_models[0][0][:20]
        worst = sorted_models[-1][0][:20]
        improvement = (distances[-1] - distances[0]) / distances[-1] * 100 if distances[-1] > 0 else 0
        ax.annotate(
            f"Best: {best}\nWorst: {worst}\nImprovement: {improvement:.0f}%",
            xy=(0.98, 0.95),
            xycoords='axes fraction',
            fontsize=9,
            ha='right',
            va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
        )
    
    plt.tight_layout()
    plt.savefig(
        output_path / f"separate_tsne_bar_{embedding_type}_{reduction_method}.png",
        dpi=150, bbox_inches='tight'
    )
    plt.close()
    
    logger.info(f"Saved separate t-SNE plots to {output_path}")


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
    single_parser.add_argument("--fsdp_devices", type=int, default=None,
                              help="Override FSDP devices (default: use config value)")
    # Target dataset arguments
    single_parser.add_argument("--target_dataset", type=str, default=None,
                              help="Target dataset name to compare against (e.g., 'bridge_v2_oxe', 'fmb')")
    single_parser.add_argument("--target_num_samples", type=int, default=None,
                              help="Number of samples from target dataset (default: same as --num_samples)")
    
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
    compare_parser.add_argument("--fsdp_devices", type=int, default=None,
                               help="Override FSDP devices (default: use config value)")
    
    # Multi-model cross-embodiment comparison (with shared t-SNE)
    compare_target_parser = subparsers.add_parser(
        "compare_target",
        help="Compare cross-embodiment transfer across multiple models with a target dataset (shared t-SNE)"
    )
    compare_target_parser.add_argument("--models", type=str, required=True,
                                       help="Comma-separated list of config:exp_name:step (step optional)")
    compare_target_parser.add_argument("--target_dataset", type=str, required=True,
                                       help="Target dataset name to compare against (e.g., 'bridge_v2_oxe', 'fmb')")
    compare_target_parser.add_argument("--num_samples", type=int, default=300,
                                       help="Number of samples from training mix per model")
    compare_target_parser.add_argument("--target_num_samples", type=int, default=100,
                                       help="Number of samples from target dataset per model")
    compare_target_parser.add_argument("--embedding_type", type=str, default="prefix_prelogits",
                                       choices=["image", "prefix_prelogits", "combined_prefix", "text_only", "per_camera", "wrist_image"])
    compare_target_parser.add_argument("--reduction_method", type=str, default="tsne",
                                       choices=["tsne", "umap", "pca"])
    compare_target_parser.add_argument("--output_dir", type=str, default="./cross_embodiment_comparison")
    compare_target_parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    compare_target_parser.add_argument("--seed", type=int, default=42)
    compare_target_parser.add_argument("--batch_size", type=int, default=None)
    compare_target_parser.add_argument("--fsdp_devices", type=int, default=None,
                                       help="Override FSDP devices (default: use config value)")
    compare_target_parser.add_argument("--separate_tsne", action="store_true",
                                       help="Run t-SNE separately for each model and plot side-by-side (in addition to shared t-SNE)")
    
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
            parser.add_argument("--fsdp_devices", type=int, default=None,
                               help="Override FSDP devices (default: use config value)")
            # Target dataset arguments
            parser.add_argument("--target_dataset", type=str, default=None,
                               help="Target dataset name to compare against (e.g., 'bridge_v2_oxe', 'fmb')")
            parser.add_argument("--target_num_samples", type=int, default=None,
                               help="Number of samples from target dataset (default: same as --num_samples)")
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
            fsdp_devices=args.fsdp_devices,
        )
    
    elif args.command == "compare_target":
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
        
        compare_cross_embodiment(
            model_configs=model_configs,
            target_dataset=args.target_dataset,
            embedding_type=args.embedding_type,
            num_samples=args.num_samples,
            target_num_samples=args.target_num_samples,
            reduction_method=args.reduction_method,
            output_dir=args.output_dir,
            split=args.split,
            seed=args.seed,
            batch_size=args.batch_size,
            separate_tsne=args.separate_tsne,
            fsdp_devices=args.fsdp_devices,
        )


def create_target_config(base_config: _config.TrainConfig, target_dataset: str) -> _config.TrainConfig:
    """Create a config with the target dataset as the data_mix.
    
    Args:
        base_config: Base training config
        target_dataset: Target dataset name (e.g., 'bridge_v2_oxe', 'fmb')
        
    Returns:
        Modified config with target dataset as data_mix
    """
    # Get the data config and modify the data_mix
    data_config = base_config.data
    
    # Create new data config with target dataset as the mix
    new_data_config = dataclasses.replace(data_config, data_mix=target_dataset)
    
    # Create new train config with the modified data config
    new_config = dataclasses.replace(base_config, data=new_data_config)
    
    return new_config


def run_single_model(args):
    """Run embedding extraction for a single model."""
    # Get config
    config = _config.get_config(args.config)
    config = dataclasses.replace(config, exp_name=args.exp_name)
    if args.batch_size:
        config = dataclasses.replace(config, batch_size=args.batch_size)
    
    logger.info(f"Config: {config.name}")
    logger.info(f"Checkpoint dir: {config.checkpoint_dir}")
    
    # Initialize runtime (with optional fsdp_devices override)
    fsdp_override = getattr(args, 'fsdp_devices', None)
    effective_fsdp = init_tpu(config, fsdp_devices_override=fsdp_override)
    
    # Load checkpoint
    train_state, data_sharding, mesh = load_checkpoint(
        config, effective_fsdp, args.checkpoint_step
    )
    
    # Create embedding extractor
    extractor = EmbeddingExtractor(train_state)
    
    # Create data loader for source/training data
    logger.info("Creating source data loader...")
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
        split=args.split,
        seed=args.seed,
    )
    
    # Extract embeddings from source data
    logger.info(f"Extracting {args.embedding_type} embeddings from source data...")
    source_result = extract_all_embeddings(
        extractor=extractor,
        data_loader=data_loader,
        embedding_type=args.embedding_type,
        num_samples=args.num_samples,
        data_sharding=data_sharding,
        mesh=mesh,
    )
    
    logger.info(f"Extracted {len(source_result.embeddings)} source embeddings")
    logger.info(f"Source datasets: {sorted(set(source_result.dataset_names))}")
    
    # Handle target dataset if specified
    target_dataset = getattr(args, 'target_dataset', None)
    target_result = None
    
    if target_dataset:
        logger.info(f"\n{'='*60}")
        logger.info(f"Extracting embeddings from target dataset: {target_dataset}")
        logger.info(f"{'='*60}")
        
        # Create config for target dataset
        target_config = create_target_config(config, target_dataset)
        
        # Create data loader for target dataset
        logger.info(f"Creating target data loader for {target_dataset}...")
        target_data_loader = _data_loader.create_data_loader(
            target_config,
            sharding=data_sharding,
            shuffle=True,
            split=args.split,
            seed=args.seed + 1,  # Different seed for variety
        )
        
        # Determine number of target samples
        target_num_samples = getattr(args, 'target_num_samples', None) or args.num_samples
        
        # Extract embeddings from target data
        logger.info(f"Extracting {args.embedding_type} embeddings from target data...")
        target_result = extract_all_embeddings(
            extractor=extractor,
            data_loader=target_data_loader,
            embedding_type=args.embedding_type,
            num_samples=target_num_samples,
            data_sharding=data_sharding,
            mesh=mesh,
        )
        
        logger.info(f"Extracted {len(target_result.embeddings)} target embeddings")
        
        # Mark target samples with a special prefix for visualization
        target_result = EmbeddingResult(
            embeddings=target_result.embeddings,
            dataset_names=[f"[TARGET] {name}" for name in target_result.dataset_names],
            sample_indices=target_result.sample_indices,
        )
    
    # Combine source and target results if target exists
    if target_result is not None:
        combined_embeddings = np.concatenate([source_result.embeddings, target_result.embeddings], axis=0)
        combined_dataset_names = source_result.dataset_names + target_result.dataset_names
        combined_sample_indices = source_result.sample_indices + [
            i + len(source_result.sample_indices) for i in target_result.sample_indices
        ]
        result = EmbeddingResult(
            embeddings=combined_embeddings,
            dataset_names=combined_dataset_names,
            sample_indices=combined_sample_indices,
        )
        logger.info(f"\nCombined total: {len(result.embeddings)} embeddings")
    else:
        result = source_result
    
    logger.info(f"Unique datasets in final result: {sorted(set(result.dataset_names))}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw embeddings
    save_name = f"embeddings_{args.embedding_type}"
    if target_dataset:
        save_name += f"_vs_{target_dataset}"
    
    np.savez(
        output_dir / f"{save_name}.npz",
        embeddings=result.embeddings,
        dataset_names=result.dataset_names,
        sample_indices=result.sample_indices,
    )
    logger.info(f"Saved raw embeddings to {output_dir / f'{save_name}.npz'}")
    
    # Reduce dimensions
    embeddings_2d = reduce_dimensions(result.embeddings, method=args.reduction_method)
    
    # Compute metrics
    metrics = compute_distance_metrics(result)
    
    # Add specific target-to-source distance metrics if target exists
    if target_dataset:
        target_mask = np.array([name.startswith("[TARGET]") for name in result.dataset_names])
        source_mask = ~target_mask
        
        if np.any(target_mask) and np.any(source_mask):
            target_centroid = np.mean(result.embeddings[target_mask], axis=0)
            source_centroid = np.mean(result.embeddings[source_mask], axis=0)
            
            metrics["target_to_source_centroid_dist"] = float(
                np.linalg.norm(target_centroid - source_centroid)
            )
            
            # Compute per-source-dataset distances
            source_names = [name for name, is_target in zip(result.dataset_names, target_mask) if not is_target]
            source_embs = result.embeddings[source_mask]
            
            unique_source_datasets = sorted(set(source_names))
            for src_dataset in unique_source_datasets:
                src_mask = np.array([n == src_dataset for n in source_names])
                if np.any(src_mask):
                    src_centroid = np.mean(source_embs[src_mask], axis=0)
                    dist = float(np.linalg.norm(target_centroid - src_centroid))
                    metrics[f"target_to_{src_dataset}_dist"] = dist
    
    # Save metrics
    metrics_path = output_dir / f"metrics_{save_name}.json"
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
    title = f"Embodiment Embeddings ({args.embedding_type}, {args.reduction_method})"
    if target_dataset:
        title += f"\nTarget: {target_dataset}"
    
    plot_embeddings(
        embeddings_2d=embeddings_2d,
        dataset_names=result.dataset_names,
        output_path=str(output_dir / f"scatter_{save_name}_{args.reduction_method}.png"),
        title=title,
    )
    
    plot_distance_heatmap(
        metrics=metrics,
        output_path=str(output_dir / f"heatmap_{save_name}.png"),
        title=f"Cross-Embodiment Distances ({args.embedding_type})",
    )
    
    # Create additional plot highlighting target vs source if target exists
    if target_dataset:
        plot_target_vs_source(
            embeddings_2d=embeddings_2d,
            dataset_names=result.dataset_names,
            output_path=str(output_dir / f"target_vs_source_{save_name}_{args.reduction_method}.png"),
            target_dataset=target_dataset,
        )
    
    logger.info("Done!")


def plot_target_vs_source(
    embeddings_2d: np.ndarray,
    dataset_names: list[str],
    output_path: str,
    target_dataset: str,
):
    """Create a scatter plot highlighting target vs source embeddings.
    
    Args:
        embeddings_2d: 2D embeddings [num_samples, 2]
        dataset_names: Dataset name for each sample
        output_path: Path to save the figure
        target_dataset: Name of the target dataset
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Separate target and source samples
    target_mask = np.array([name.startswith("[TARGET]") for name in dataset_names])
    source_mask = ~target_mask
    
    # Plot source samples in gray
    ax.scatter(
        embeddings_2d[source_mask, 0],
        embeddings_2d[source_mask, 1],
        c='lightgray',
        label='Source (training mix)',
        alpha=0.4,
        s=20,
    )
    
    # Plot target samples in red with larger markers
    ax.scatter(
        embeddings_2d[target_mask, 0],
        embeddings_2d[target_mask, 1],
        c='red',
        label=f'Target ({target_dataset})',
        alpha=0.8,
        s=50,
        edgecolors='darkred',
        linewidths=0.5,
    )
    
    # Compute and plot centroids
    if np.any(source_mask):
        source_centroid = np.mean(embeddings_2d[source_mask], axis=0)
        ax.scatter(
            source_centroid[0], source_centroid[1],
            c='blue', marker='X', s=200, label='Source centroid',
            edgecolors='darkblue', linewidths=2,
        )
    
    if np.any(target_mask):
        target_centroid = np.mean(embeddings_2d[target_mask], axis=0)
        ax.scatter(
            target_centroid[0], target_centroid[1],
            c='red', marker='X', s=200, label='Target centroid',
            edgecolors='darkred', linewidths=2,
        )
        
        # Draw line between centroids
        if np.any(source_mask):
            ax.plot(
                [source_centroid[0], target_centroid[0]],
                [source_centroid[1], target_centroid[1]],
                'k--', linewidth=2, alpha=0.5,
            )
            # Add distance annotation
            dist = np.linalg.norm(target_centroid - source_centroid)
            mid_point = (source_centroid + target_centroid) / 2
            ax.annotate(
                f'd={dist:.2f}',
                xy=mid_point,
                fontsize=10,
                ha='center',
                va='bottom',
            )
    
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title(f"Target ({target_dataset}) vs Source Embeddings")
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved target vs source plot to {output_path}")


def initialize_distributed():
    """Initialize JAX distributed system at program start.
    
    This must be called before any other JAX operations.
    Orbax checkpoint manager requires this even on single-host setups.
    """
    # Check if multi-host environment variables are set
    has_coordinator = os.environ.get("JAX_COORDINATOR_ADDRESS") is not None
    has_slurm = os.environ.get("SLURM_JOB_ID") is not None
    has_tpu_worker = os.environ.get("TPU_WORKER_ID") is not None
    has_cloud_tpu = os.environ.get("CLOUD_TPU_TASK_ID") is not None
    
    is_multi_host = has_coordinator or has_slurm or has_tpu_worker or has_cloud_tpu
    
    try:
        if is_multi_host:
            # Multi-host: let JAX auto-detect from environment
            jax.distributed.initialize()
            logger.info("Initialized JAX distributed system (multi-host)")
        else:
            # Single-host: use local coordinator
            # Find a free port for the coordinator
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))
                port = s.getsockname()[1]
            
            jax.distributed.initialize(
                coordinator_address=f"localhost:{port}",
                num_processes=1,
                process_id=0,
            )
            logger.info(f"Initialized JAX distributed system (single-host mode, port={port})")
    except Exception as e:
        # May fail if already initialized
        logger.warning(f"JAX distributed initialization note: {e}")
        logger.info("Continuing (may already be initialized or not needed)")


if __name__ == "__main__":
    # Initialize distributed BEFORE any JAX operations
    initialize_distributed()
    main()
