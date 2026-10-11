"""
Draw the learning curves from recordings/training_log.json (written by train.py).

    python3 -m rl.plot_training

Saves recordings/learning_curves.png.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")  # no window needed: we save an image directly
import matplotlib.pyplot as plt
import numpy as np

from rl.gym_env import GRID_SIZE, REWARD_DROP_NEAR, REWARD_PICK_UP, REWARD_STORE

RECORDINGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "recordings")

# Best possible reward for one game, ignoring the small per-turn cost
MAX_REWARD = REWARD_PICK_UP + REWARD_DROP_NEAR + REWARD_STORE


def moving_average(values, window):
    """Average of each value with the (window - 1) values before it. Smooths a noisy curve."""
    values = np.asarray(values, dtype=float)
    return np.convolve(values, np.ones(window) / window, mode="valid")


def main():
    with open(os.path.join(RECORDINGS_DIR, "training_log.json")) as f:
        log = json.load(f)

    n_games = len(log["reward"])
    window = min(200, n_games)          # smaller window if there are only a few games
    games = np.arange(n_games)
    smoothed_x = games[window - 1:]     # the smoothed curve starts once a full window is available

    config = log.get("config", {})      # older logs have no config
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    fig.suptitle(
        f"RoboRelay - tabular Q-learning (Carrier + Placer, {GRID_SIZE}x{GRID_SIZE} grid)\n"
        f"{n_games} games, seed {config.get('seed', '?')}",
        fontsize=11,
    )

    # 1. Reward per game
    axes[0].plot(games, log["reward"], color="lightgray", linewidth=0.5, label="per game")
    axes[0].plot(smoothed_x, moving_average(log["reward"], window), color="tab:blue",
                 label=f"moving average ({window} games)")
    axes[0].axhline(MAX_REWARD, color="tab:green", linestyle="--", linewidth=1,
                    label=f"best possible ({MAX_REWARD:g})")
    axes[0].set_ylabel("Reward")
    axes[0].legend(loc="lower right")

    # 2. Success rate, with epsilon on a second axis (right side)
    axes[1].plot(smoothed_x, moving_average(log["success"], window) * 100, color="tab:orange",
                 label="success rate")
    axes[1].set_ylabel("Success (%)")
    axes[1].set_ylim(0, 105)
    eps_ax = axes[1].twinx()
    eps_ax.plot(games, log["epsilon"], color="tab:purple", linestyle=":", label="epsilon (exploration)")
    eps_ax.set_ylabel("Epsilon")
    eps_ax.set_ylim(0, 1.05)
    handles, labels = axes[1].get_legend_handles_labels()
    eps_handles, eps_labels = eps_ax.get_legend_handles_labels()
    axes[1].legend(handles + eps_handles, labels + eps_labels, loc="center right")

    # 3. Game length
    axes[2].plot(smoothed_x, moving_average(log["turns"], window), color="tab:red")
    axes[2].set_ylabel("Turns per game")
    axes[2].set_xlabel("Game")

    for ax in axes:
        ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(RECORDINGS_DIR, "learning_curves.png")
    fig.savefig(out_path, dpi=120)
    print(f"Curves saved: {out_path}")


if __name__ == "__main__":
    main()