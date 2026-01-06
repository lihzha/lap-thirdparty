import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from openpi_cot.training import config as train_config


def test_plot_delayed_ema_schedule(tmp_path):
    config = train_config.TrainConfig(
        name="test",
        exp_name="test",
        num_train_steps=30_000,
        ema_decay=0.999,
        ema_schedule_choice=train_config.EmaScheduleChoice(
            kind="cosine_delayed",
            start_step=10_000,
        ),
    )

    steps = np.arange(config.num_train_steps, dtype=np.int32)
    decay, enabled = config.get_ema_decay_for_step(steps)
    decay = np.asarray(decay)
    enabled = np.asarray(enabled)

    assert decay.shape == steps.shape
    assert enabled.shape == steps.shape
    assert np.all(decay[: config.ema_schedule_choice.start_step] == 0.0)
    assert np.isclose(decay[-1], config.ema_decay, rtol=0.0, atol=5e-4)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, decay, linewidth=2)
    ax.set_title("Cosine-delayed EMA schedule")
    ax.set_xlabel("Step")
    ax.set_ylabel("EMA decay")
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()

    output_path = tmp_path / "ema_schedule.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    assert output_path.exists()
