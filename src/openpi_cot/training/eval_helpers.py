"""Evaluate token accuracy and/or rollout performance on a checkpoint using validation data."""

import dataclasses
import logging
import math

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from openpi.models import model as _model
import openpi.shared.array_typing as at
import tqdm_loggable.auto as tqdm
import wandb

from openpi_cot.models.model_adapter import CoTObservation
import openpi_cot.training.config as _config
import openpi_cot.training.mh_sharding as sharding
import openpi_cot.training.utils as training_utils
import openpi_cot.training.vis_tools as vis_tools


class TokenAccuracyEvaluator:
    def __init__(self, config: _config.TrainConfig):
        self.config = config

    @at.typecheck
    def __call__(
        self,
        rng: at.KeyArrayLike,
        state: training_utils.TrainState,
        batch: tuple[CoTObservation, _model.Actions],
    ) -> dict[str, at.Array]:
        model = nnx.merge(state.model_def, state.params)
        model.eval()

        @at.typecheck
        def eval_fn(
            model: _model.BaseModel,
            rng: at.KeyArrayLike,
            observation: CoTObservation,
            actions: _model.Actions,
        ):
            loss, metrics = model.compute_loss(rng, observation, actions, train=False, verbose_mode=True)
            return loss, metrics

        eval_rng = jax.random.fold_in(rng, state.step)
        observation, actions = batch
        loss, metrics = eval_fn(model, eval_rng, observation, actions)
        print(metrics.keys())

        info = {
            "loss": loss,
            "token_accuracy": metrics["token_accuracy"],
            "critical_token_accuracy": metrics["critical_token_accuracy"],
            "number_token_accuracy": metrics["number_token_accuracy"],
            "direction_token_accuracy": metrics["direction_token_accuracy"],
            "per_token_loss": metrics["per_token_loss"],
            "labels": metrics["labels"],
        }
        return info


class RolloutEvaluator:
    def __init__(self, config: _config.TrainConfig):
        self.config = config

    @at.typecheck
    def __call__(
        self,
        rng: at.KeyArrayLike,
        state: training_utils.TrainState,
        batch: tuple[CoTObservation, _model.Actions],
    ) -> jax.Array:
        """Sample language action tokens for rollout evaluation.

        Returns:
            id_buf: Sampled token IDs [batch, seq_len, 1]
            t_final: Final sequence length
        """
        model = nnx.merge(state.model_def, state.params)
        model.eval()

        # Prepare eval batch (remove language action from ground truth)
        observation, _ = batch

        # Sample language action tokens
        output_tokens = model.sample_tokens(rng, observation)

        return output_tokens


def eval_checkpoint(
    train_state,
    config: _config.TrainConfig,
    mesh,
    data_sharding,
    replicated_sharding,
    data_loader,
    eval_rng,
    train_state_sharding,
):
    # # Use EMA params for evaluation if available (matches training checkpoint layout)
    if train_state.ema_params is not None and config.eval_use_ema:
        logging.info("Using EMA params for evaluation")
        train_state = dataclasses.replace(train_state, params=train_state.ema_params)

    # Determine number of evaluation batches
    num_eval_batches = config.num_eval_batches
    if num_eval_batches is None:
        if getattr(config.data, "val_max_samples", None):
            process_count = getattr(jax, "process_count", lambda: 1)()
            local_bs = max(1, config.batch_size // process_count)
            num_eval_batches = math.ceil(config.data.val_max_samples / local_bs)
        else:
            # Default to 1000 batches if not specified
            num_eval_batches = 1000

    logging.info(f"Evaluating over {num_eval_batches} batches with mode: {config.eval_mode}")

    results = {}

    logging.info("Running rollout evaluation...")
    rollout_results = evaluate_rollout(
        config,
        eval_rng,
        train_state,
        train_state_sharding,
        data_loader,
        mesh,
        data_sharding,
        replicated_sharding,
        num_eval_batches,
    )
    results.update(rollout_results)

    logging.info("Running token accuracy evaluation...")
    token_results = evaluate_token_accuracy(
        config,
        eval_rng,
        train_state,
        train_state_sharding,
        data_loader,
        mesh,
        data_sharding,
        replicated_sharding,
        num_eval_batches,
    )
    results.update(token_results)

    logging.info("=" * 80)
    logging.info("EVALUATION RESULTS")
    logging.info("=" * 80)
    for key, value in results.items():
        logging.info(f"{key:40s}: {value}")
    logging.info("=" * 80)

    # Log to wandb
    if jax.process_index() == 0 and config.wandb_enabled:
        wandb.log(results, step=int(train_state.step))
        wandb.summary.update(results)

    return results


def evaluate_token_accuracy(
    config: _config.TrainConfig,
    eval_rng: at.KeyArrayLike,
    train_state: training_utils.TrainState,
    train_state_sharding,
    data_loader,
    mesh,
    data_sharding,
    replicated_sharding,
    num_eval_batches: int,
) -> dict[str, float]:
    """Evaluate token accuracy."""
    evaluator = TokenAccuracyEvaluator(config)
    peval_step = jax.jit(
        evaluator,
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
    )

    data_iter = iter(data_loader)

    pbar = tqdm.tqdm(
        range(num_eval_batches),
        total=num_eval_batches,
        dynamic_ncols=True,
        desc="Token accuracy evaluation",
        disable=(jax.process_index() != 0),
    )

    number_token_losses = []
    direction_token_losses = []
    other_token_losses = []

    number_token_accuracies = []
    direction_token_accuracies = []
    all_token_accuracies = []

    with sharding.set_mesh(mesh):
        for batch_idx in pbar:
            try:
                batch = next(data_iter)
            except StopIteration:
                logging.info(f"Reached end of dataset at batch {batch_idx}")
                break

            eval_info = peval_step(eval_rng, train_state, batch)
            # Bring sharded arrays to host memory for local computation.
            per_token_loss = jnp.asarray(training_utils.to_local_array(eval_info["per_token_loss"]))
            number_mask = jnp.asarray(training_utils.to_local_array(batch[0].number_token_mask))[:, 1:]
            direction_mask = jnp.asarray(training_utils.to_local_array(batch[0].direction_token_mask))[:, 1:]
            tokenized_langact_mask = jnp.asarray(training_utils.to_local_array(batch[0].tokenized_langact_mask))[:, 1:]

            def _masked_mean(mask):
                mask_sum = jnp.sum(mask)
                return jnp.sum(per_token_loss * mask) / mask_sum

            number_token_loss = _masked_mean(number_mask)
            direction_token_loss = _masked_mean(direction_mask)
            other_token_mask = jnp.logical_and(
                tokenized_langact_mask,
                jnp.logical_not(
                    jnp.logical_or(
                        number_mask,
                        direction_mask,
                    )
                ),
            )
            other_token_loss = _masked_mean(other_token_mask)
            number_token_losses.append(number_token_loss)
            direction_token_losses.append(direction_token_loss)
            other_token_losses.append(other_token_loss)

            number_token_accuracies.append(
                jnp.asarray(training_utils.to_local_array(eval_info["number_token_accuracy"]))
            )
            direction_token_accuracies.append(
                jnp.asarray(training_utils.to_local_array(eval_info["direction_token_accuracy"]))
            )
            all_token_accuracies.append(jnp.asarray(training_utils.to_local_array(eval_info["token_accuracy"])))

    return {
        "eval/number_token_loss": float(jnp.mean(jnp.array(number_token_losses))),
        "eval/direction_token_loss": float(jnp.mean(jnp.array(direction_token_losses))),
        "eval/other_token_loss": float(jnp.mean(jnp.array(other_token_losses))),
        "eval/number_token_accuracy": float(jnp.mean(jnp.array(number_token_accuracies))),
        "eval/direction_token_accuracy": float(jnp.mean(jnp.array(direction_token_accuracies))),
        "eval/token_accuracy": float(jnp.mean(jnp.array(all_token_accuracies))),
    }


def evaluate_rollout(
    config: _config.TrainConfig,
    eval_rng: at.KeyArrayLike,
    train_state: training_utils.TrainState,
    train_state_sharding,
    data_loader,
    mesh,
    data_sharding,
    replicated_sharding,
    num_eval_batches: int,
    max_logged_imgs: int = 100,
) -> dict[str, float]:
    """Evaluate rollout performance (language action prediction accuracy)."""
    evaluator = RolloutEvaluator(config)
    # Use data sharding so each host only supplies its local batch shard.
    peval_step = jax.jit(
        evaluator,
        in_shardings=(replicated_sharding, train_state_sharding, replicated_sharding),
        out_shardings=(replicated_sharding),
    )
    # peval_step = evaluator
    # Get tokenizer for decoding
    tokenizer = data_loader.tokenizer

    data_iter = iter(data_loader)
    images_to_log = []
    num_logged_imgs = 0

    pbar = tqdm.tqdm(
        range(num_eval_batches),
        total=num_eval_batches,
        dynamic_ncols=True,
        desc="Rollout evaluation",
        disable=(jax.process_index() != 0),
    )

    with sharding.set_mesh(mesh):
        for batch_idx in pbar:
            try:
                batch = next(data_iter)
            except StopIteration:
                logging.info(f"Reached end of dataset at batch {batch_idx}")
                break

            # Prepare eval batch (remove language actions) and replicate for JIT
            eval_batch = vis_tools.prepare_eval_batch(batch)
            # Replicate the batch to match expected sharding
            eval_batch_replicated = jax.device_put(eval_batch, replicated_sharding)

            # Run rollout evaluation
            output_tokens = peval_step(eval_rng, train_state, eval_batch_replicated)

            # Process results on host
            # if jax.process_index() == 0:
            # Bring sharded outputs to the host before decoding/logging.
            # output_tokens_local = training_utils.to_local_array(output_tokens)
            output_tokens_local = output_tokens
            k_local = min(config.batch_size, batch[0].state.shape[0])
            gt_texts, pred_texts = vis_tools.eval_step(batch, output_tokens_local, tokenizer, k_local)

            base_img = training_utils.to_local_array(eval_batch_replicated[0].images.get("base_0_rgb"))
            if base_img is None:
                continue
            imgs_local = training_utils.to_local_array(base_img)
            num_available = imgs_local.shape[0]
            num_entries = min(k_local, len(gt_texts), len(pred_texts), num_available)
            if num_entries == 0:
                continue
            imgs_to_log = imgs_local[:num_entries]

            for i in range(num_entries):
                gt_text = gt_texts[i]
                pred_text = pred_texts[i]
                img = imgs_to_log[i]

                img_to_log = vis_tools.create_rollout_visualization(
                    img,
                    gt_text,
                    pred_text,
                )
                images_to_log.append(wandb.Image(img_to_log))
                num_logged_imgs += 1
            if num_logged_imgs >= max_logged_imgs:
                break

    return {"eval/rollout": images_to_log}
