import logging

import tensorflow as tf

from src.data.dataset import make_interleaved_dataset
from src.data.oxe import make_oxe_dataset_kwargs_and_weights
from src.utils.monitor import log_execution_time

tf.config.set_visible_devices([], "GPU")
log = logging.getLogger(__name__)


class RLDSInterleavedDataloader:
    @log_execution_time(log)
    def __init__(self, config, train=False, batch_size=64):
        dataset_kwargs_list, sample_weights = make_oxe_dataset_kwargs_and_weights(
            config.dataset_mix,
            config.data_path,
            load_proprio=config.load_proprio,
            load_camera_views=("primary",),
            skip_norm=True,
        )

        rlds_config = dict(
            dataset_kwargs_list=dataset_kwargs_list,
            sample_weights=sample_weights,
            train=train,
            split=config.get("split", "train"),
            balance_weights=True,
            traj_transform_kwargs=dict(
                # goal_relabeling_strategy="uniform",   # no neeed for goal relabeling
                window_size=config.window_size,
                action_horizon=config.action_horizon,
                # subsample_length=100,
                skip_unlabeled=config.skip_unlabeled,  # skip ones without language annotation
            ),
            frame_transform_kwargs=dict(
                resize_size=dict(
                    primary=(224, 224),
                    wrist=(224, 224),
                ),
                num_parallel_calls=config.num_parallel_calls,
            ),
            traj_transform_threads=config.traj_transform_threads,
            traj_read_threads=config.traj_read_threads,
            apply_trajwise_image_aug=False,
        )

        self.dataset = make_interleaved_dataset(**rlds_config, batch_size=batch_size)

    def __iter__(self):
        yield from self.dataset.as_numpy_iterator()

    def __len__(self):
        raise NotImplementedError("len() not implemented for RLDSInterleavedDataloader")
