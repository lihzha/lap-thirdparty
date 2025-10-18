"""Mock training script for testing training infrastructure without expensive components.

This script tests the training loop with:
- Fake/minimal model (just parameters, no real computation)
- Fake dataset (random data generation)
- Fake loss calculation (deterministic with some high-loss samples)
- Real training infrastructure (optimizer, checkpointing, logging, hard examples)

Usage:
    # Single host
    python scripts/test_training_mock.py --num_train_steps 100 --log_interval 10

    # Multi-host simulation (if on TPU/multi-GPU)
    python scripts/test_training_mock.py --num_train_steps 100 --fsdp_devices 2
"""

import dataclasses
import logging
import os
import sys
from pathlib import Path
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from openpi_cot.models.adapters.model_adapter import CoTObservation
from openpi_cot.training import utils as training_utils
import openpi_cot.training.vis_tools as vis_tools
import openpi_cot.training.checkpoints as _checkpoints


# ============================================================================
# Mock Components
# ============================================================================


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size

    def encode(self, text: str) -> np.ndarray:
        """Encode text to token IDs."""
        # Simple mock: hash the text to get consistent tokens
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        num_tokens = (hash_val % 50) + 10  # 10-60 tokens
        rng = np.random.default_rng(seed=hash_val)
        return rng.integers(0, self.vocab_size, size=num_tokens, dtype=np.int32)

    def decode(self, tokens: np.ndarray) -> str:
        """Decode token IDs to text."""
        tokens = np.asarray(tokens).flatten()
        # Mock: create readable output based on tokens
        if len(tokens) == 0:
            return ""

        # Generate mock action text
        movements = ["move right", "move left", "move forward", "move backward", "move up", "move down"]
        distances = ["5cm", "10cm", "15cm", "20cm"]

        # Use token values to deterministically pick movements
        num_moves = min(3, (len(tokens) % 3) + 1)
        parts = []
        for i in range(num_moves):
            token_idx = int(tokens[i % len(tokens)])
            movement = movements[token_idx % len(movements)]
            distance = distances[token_idx % len(distances)]
            parts.append(f"{movement} {distance}")

        return " and ".join(parts) + f" (len={len(tokens)})"


class MockDataset:
    """Mock dataset that generates random batches."""

    def __init__(
        self,
        batch_size: int = 16,
        image_size: tuple[int, int] = (224, 224),
        seq_len: int = 128,
        seed: int = 42,
        num_cameras: int = 2,
    ):
        self.batch_size = batch_size
        self.image_size = image_size
        self.seq_len = seq_len
        self.seed = seed
        self.num_cameras = num_cameras
        self.tokenizer = MockTokenizer()
        self._step = 0

    def __iter__(self):
        return self

    def __next__(self) -> tuple[CoTObservation, Any]:
        """Generate a mock batch."""
        # Use step-based seed for reproducibility
        rng = np.random.default_rng(seed=self.seed + self._step)
        self._step += 1

        # Generate mock images (normalized to [-1, 1])
        images = {}
        image_masks = {}
        for cam_idx in range(self.num_cameras):
            cam_name = f"cam_{cam_idx}"
            # Generate images in uint8 first, then normalize
            img_uint8 = rng.integers(0, 256,
                                      (self.batch_size, *self.image_size, 3),
                                      dtype=np.uint8)
            # Normalize to [-1, 1]
            images[cam_name] = (img_uint8.astype(np.float32) / 127.5) - 1.0
            image_masks[cam_name] = np.ones(self.batch_size, dtype=bool)

        # Generate mock tokenized sequences
        tokenized_prompt = rng.integers(0, 32000,
                                         (self.batch_size, self.seq_len),
                                         dtype=np.int32)

        # Create masks (80% of tokens are valid)
        tokenized_prompt_mask = rng.random((self.batch_size, self.seq_len)) > 0.2

        # Language action mask (20% of tokens are language actions)
        tokenized_langact_mask = rng.random((self.batch_size, self.seq_len)) < 0.2
        tokenized_langact_mask = tokenized_langact_mask & tokenized_prompt_mask

        # Token AR mask (autoregressive mask)
        token_ar_mask = rng.random((self.batch_size, self.seq_len)) > 0.1

        # Token loss mask (which tokens contribute to loss)
        token_loss_mask = tokenized_prompt_mask.copy()

        # Sample mask (all samples valid)
        sample_mask = np.ones(self.batch_size, dtype=bool)

        # Robot state
        state = rng.random((self.batch_size, 7), dtype=np.float32)

        obs = CoTObservation(
            images=images,
            image_masks=image_masks,
            state=state,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
            tokenized_langact_mask=tokenized_langact_mask,
            token_ar_mask=token_ar_mask,
            token_loss_mask=token_loss_mask,
            sample_mask=sample_mask,
        )

        # Mock actions (not really used in this test)
        actions = type('Actions', (), {
            'world_vector': rng.random((self.batch_size, 7), dtype=np.float32),
            'rotation_vector': rng.random((self.batch_size, 6), dtype=np.float32),
            'gripper_command': rng.random((self.batch_size, 1), dtype=np.float32),
        })()

        return obs, actions


class MockModel(nnx.Module):
    """Mock model with minimal parameters for testing."""

    def __init__(self, rng_key, hidden_dim: int = 128, num_layers: int = 2):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Create minimal parameters as nnx.Param
        self.layers = {}
        for i in range(num_layers):
            key1, key2, rng_key = jax.random.split(rng_key, 3)
            self.layers[f'layer_{i}'] = {
                'weight': nnx.Param(jax.random.normal(key1, (hidden_dim, hidden_dim))),
                'bias': nnx.Param(jax.random.normal(key2, (hidden_dim,))),
            }

    def compute_loss(
        self,
        rng_key,
        observation: CoTObservation,
        actions: Any,
        train: bool = True,
    ):
        """Compute fake loss with some high-loss samples.

        Returns:
            per_sample_loss: Array of shape (batch_size,)
            token_accuracy: Scalar accuracy
            critical_token_accuracy: Scalar accuracy for critical tokens
            metrics: Dict of additional metrics
        """
        batch_size = observation.tokenized_prompt.shape[0]

        # Generate deterministic losses based on batch content
        # Use hash of tokenized_prompt to get deterministic but varied losses
        prompt_hash = jnp.sum(observation.tokenized_prompt, axis=1)

        # Base loss in range [0.5, 2.0]
        base_loss = 0.5 + (prompt_hash % 150) / 100.0

        # Make some samples have high loss (simulating hard examples)
        # Every 5th sample gets high loss
        high_loss_mask = (prompt_hash % 5) == 0
        high_loss_values = 3.0 + (prompt_hash % 700) / 100.0  # 3.0-10.0

        per_sample_loss = jnp.where(high_loss_mask, high_loss_values, base_loss)

        # Add some noise if training
        if train:
            noise = jax.random.normal(rng_key, (batch_size,)) * 0.1
            per_sample_loss = per_sample_loss + noise

        # Mock accuracies
        token_accuracy = 0.7 + jax.random.uniform(rng_key, ()) * 0.2
        critical_token_accuracy = 0.6 + jax.random.uniform(rng_key, ()) * 0.2

        # Mock additional metrics
        metrics = {
            'action_loss': per_sample_loss * 0.6,
            'reasoning_loss': per_sample_loss * 0.4,
            'action_accuracy': token_accuracy * 1.1,
        }

        return per_sample_loss, token_accuracy, critical_token_accuracy, metrics


# ============================================================================
# Training Configuration
# ============================================================================


@dataclasses.dataclass
class MockTrainConfig:
    """Minimal training configuration for testing."""

    # Experiment
    name: str = "mock_training_test"
    exp_name: str = "mock_exp"
    project_name: str = "mock_project"
    seed: int = 42

    # Training
    num_train_steps: int = 100
    batch_size: int = 16
    log_interval: int = 10
    save_interval: int = 50

    # Checkpointing
    checkpoint_dir: epath.Path = epath.Path("/tmp/mock_training_checkpoints")
    keep_period: int = 50
    overwrite: bool = True
    resume: bool = False
    checkpoint_async_enable: bool = False
    checkpoint_async_timeout_secs: int = 60
    checkpoint_max_retries: int = 3
    checkpoint_retry_delay_secs: int = 5
    checkpoint_retry_backoff: float = 2.0
    checkpoint_fallback_to_sync: bool = True

    # Distributed
    fsdp_devices: int = 1

    # Logging
    wandb_enabled: bool = False

    # Hard examples
    track_hard_examples: bool = True
    max_hard_examples_buffer: int = 50
    max_hard_examples_log: int = 10
    hard_example_log_interval: int = 50

    def __post_init__(self):
        self.checkpoint_dir = epath.Path(self.checkpoint_dir)


# ============================================================================
# Training Functions
# ============================================================================


def init_logging():
    """Initialize logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def create_mock_train_state(config: MockTrainConfig, rng_key) -> training_utils.TrainState:
    """Create a mock training state."""
    # Create mock model
    model = MockModel(rng_key, hidden_dim=128, num_layers=2)

    # Create optimizer
    learning_rate = 1e-3
    tx = optax.adam(learning_rate)

    # Get params and graphdef using nnx
    graphdef, state = nnx.split(model)
    params = nnx.state(model)
    opt_state = tx.init(params)

    return training_utils.TrainState(
        step=0,
        params=params,
        model_def=graphdef,
        tx=tx,
        opt_state=opt_state,
        ema_decay=None,
        ema_params=None,
    )


def train_step(
    rng_key,
    state: training_utils.TrainState,
    batch: tuple[CoTObservation, Any],
    model: MockModel,
) -> tuple[training_utils.TrainState, dict[str, jnp.ndarray]]:
    """Execute one training step."""

    def loss_fn(params, rng_key, observation, actions):
        # Mock: use params to compute loss (just for demo)
        per_sample_loss, token_accuracy, critical_token_accuracy, metrics = model.compute_loss(
            rng_key, observation, actions, train=True
        )
        return jnp.mean(per_sample_loss), (per_sample_loss, token_accuracy, critical_token_accuracy, metrics)

    train_rng = jax.random.fold_in(rng_key, state.step)
    observation, actions = batch

    # Compute loss and gradients
    grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
    (loss, (per_sample_loss, token_accuracy, critical_token_accuracy, loss_metrics)), grads = grad_fn(
        state.params, train_rng, observation, actions
    )

    # Apply updates
    updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)

    # Create new state
    new_state = dataclasses.replace(
        state,
        step=state.step + 1,
        params=new_params,
        opt_state=new_opt_state,
    )

    # Compute norms
    grad_norm = optax.global_norm(grads)
    param_norm = optax.global_norm(new_params)

    info = {
        'loss': loss,
        'per_sample_loss': per_sample_loss,
        'grad_norm': grad_norm,
        'param_norm': param_norm,
        'token_accuracy': token_accuracy,
        'critical_token_accuracy': critical_token_accuracy,
    }

    # Add loss components
    for key, value in loss_metrics.items():
        if key.endswith('_loss'):
            info[key] = jnp.mean(value)
        else:
            info[key] = value

    return new_state, info


def mock_training_loop(config: MockTrainConfig):
    """Main mock training loop."""
    init_logging()

    logging.info("="*80)
    logging.info("MOCK TRAINING TEST")
    logging.info("="*80)
    logging.info(f"Configuration:")
    logging.info(f"  Steps: {config.num_train_steps}")
    logging.info(f"  Batch size: {config.batch_size}")
    logging.info(f"  Log interval: {config.log_interval}")
    logging.info(f"  Track hard examples: {config.track_hard_examples}")
    logging.info(f"  Checkpoint dir: {config.checkpoint_dir}")
    logging.info("="*80)

    # Initialize RNG
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    # Create mock dataset
    dataset = MockDataset(
        batch_size=config.batch_size,
        image_size=(224, 224),
        seq_len=128,
        seed=config.seed,
    )
    data_iter = iter(dataset)

    # Create mock model and training state
    logging.info("Initializing mock model and training state...")
    model = MockModel(init_rng, hidden_dim=128, num_layers=2)
    train_state = create_mock_train_state(config, init_rng)

    # Initialize checkpoint manager
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
        async_timeout_secs=config.checkpoint_async_timeout_secs,
        async_enable=config.checkpoint_async_enable,
    )

    # Initialize hard example tracker
    hard_example_tracker = None
    if config.track_hard_examples:
        logging.info("Initializing hard example tracker...")
        hard_example_tracker = vis_tools.HardExampleTracker(
            tokenizer=dataset.tokenizer,
            max_hard_examples=config.max_hard_examples_buffer,
            resize_hw=(128, 128),
        )

    # Training loop
    logging.info("Starting training loop...")
    start_step = int(train_state.step)

    for step in range(start_step, config.num_train_steps):
        # Get batch
        batch = next(data_iter)

        # Training step
        train_state, info = train_step(train_rng, train_state, batch, model)

        # Convert info to numpy for logging
        info_np = jax.device_get(info)
        per_sample_loss = info_np['per_sample_loss']

        # Track hard examples
        if hard_example_tracker is not None:
            hard_example_tracker.update(per_sample_loss)

            # Periodically add examples with images
            if step % config.log_interval == 0:
                # Convert batch to host (numpy) for visualization
                host_batch = jax.tree.map(lambda x: np.asarray(x), batch)

                hard_example_tracker.add_local_examples(
                    step_idx=step,
                    host_batch_local=host_batch,
                    local_losses=per_sample_loss,
                    global_idx_base=0,  # Single host
                    process_idx=jax.process_index(),
                )

        # Logging
        if step % config.log_interval == 0:
            loss_val = info_np['loss']
            grad_norm = info_np['grad_norm']
            param_norm = info_np['param_norm']
            token_acc = info_np['token_accuracy']

            logging.info(
                f"Step {step:4d}: loss={loss_val:.4f}, "
                f"grad_norm={grad_norm:.4f}, param_norm={param_norm:.4f}, "
                f"token_acc={token_acc:.4f}, "
                f"max_per_sample_loss={np.max(per_sample_loss):.4f}"
            )

            # Log hard examples
            if hard_example_tracker is not None and step % config.hard_example_log_interval == 0 and step > 0:
                payload = hard_example_tracker.log_if_ready(step_idx=step)
                if payload is not None:
                    entries = payload.get('entries', [])
                    threshold = payload.get('quantile_threshold', 0.0)
                    total_samples = payload.get('total_samples', 0)

                    logging.info(
                        f"  Hard examples: {len(entries)} entries, "
                        f"threshold={threshold:.4f}, "
                        f"total_samples={total_samples}"
                    )

                    # Show top 3
                    for i, entry in enumerate(entries[:3]):
                        logging.info(
                            f"    #{i+1}: loss={entry['loss']:.4f}, "
                            f"step={entry['step']}, "
                            f"lang_action={entry.get('language_action', 'N/A')[:50]}..."
                        )

        # Checkpointing
        if step % config.save_interval == 0 and step > 0:
            logging.info(f"Saving checkpoint at step {step}...")
            # Note: We'd need to adapt save_state for mock dataset
            # For now, just log
            logging.info(f"  (Checkpoint saving skipped in mock mode)")

    logging.info("="*80)
    logging.info("Training complete!")
    logging.info("="*80)

    # Final statistics
    if hard_example_tracker is not None:
        logging.info(f"Hard example buffer size: {len(hard_example_tracker._hard_example_buffer)}")
        logging.info(f"Hard example keys tracked: {len(hard_example_tracker._hard_example_keys)}")


# ============================================================================
# CLI
# ============================================================================


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="Mock training test")
    parser.add_argument("--num_train_steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=50, help="Checkpoint save interval")
    parser.add_argument("--fsdp_devices", type=int, default=1, help="Number of FSDP devices")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint_dir", type=str, default="/tmp/mock_training_checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--no_hard_examples", action="store_true", help="Disable hard example tracking")
    parser.add_argument("--max_hard_examples_buffer", type=int, default=50,
                        help="Max hard examples to buffer")
    parser.add_argument("--max_hard_examples_log", type=int, default=10,
                        help="Max hard examples to log")
    parser.add_argument("--hard_example_log_interval", type=int, default=50,
                        help="Interval for logging hard examples")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    config = MockTrainConfig(
        num_train_steps=args.num_train_steps,
        batch_size=args.batch_size,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        fsdp_devices=args.fsdp_devices,
        seed=args.seed,
        checkpoint_dir=epath.Path(args.checkpoint_dir),
        track_hard_examples=not args.no_hard_examples,
        max_hard_examples_buffer=args.max_hard_examples_buffer,
        max_hard_examples_log=args.max_hard_examples_log,
        hard_example_log_interval=args.hard_example_log_interval,
    )

    try:
        mock_training_loop(config)
    except KeyboardInterrupt:
        logging.info("\nTraining interrupted by user")
    except Exception as e:
        logging.error(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
