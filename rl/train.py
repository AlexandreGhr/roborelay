"""
Train the tabular Q-learning agent.

    python3 -m rl.train                              # default: 20000 games, seed 0
    python3 -m rl.train --episodes 5000 --seed 1

Saves in recordings/:
    q_agent.pkl            the trained agent (used by evaluate.py and watch_agent.py)
    training_log.json      one line of stats per game + the settings used (for plot_training.py)
    training_history.json  every action of 11 games spread over training (first to last)
"""

import argparse
import json
import os

import numpy as np

from rl.gym_env import RoboRelayEnv
from rl.q_learning import QLearningAgent

RECORDINGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "recordings")


def run_episode(env, agent, record=False):
    """Play one game and learn from every step.

    Returns (total_reward, details). details lists every action and reward
    when record=True, otherwise it is None.
    """
    obs, _ = env.reset(randomize=True)
    details = None
    if record:
        details = {
            "carrier_spawn": list(env.carrier.position),
            "placer_spawn": list(env.placer.position),
            "objects_start": [list(pos) for pos in env.world.current_object_positions],
            "actions": [],
            "rewards": [],
        }

    total_reward = 0.0
    done = False
    while not done:
        carrier_action, placer_action = agent.choose_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(np.array([carrier_action, placer_action]))

        # Pass terminated only (not truncated): a time limit is not a real end of the game
        agent.update(obs, carrier_action, placer_action, reward, next_obs, terminated)

        total_reward += reward
        if record:
            details["actions"].append([int(carrier_action), int(placer_action)])
            details["rewards"].append(round(float(reward), 3))

        obs = next_obs
        done = terminated or truncated

    return total_reward, details


def save_json(data, filename):
    path = os.path.join(RECORDINGS_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=1)
    return path


def main():
    parser = argparse.ArgumentParser(description="Train the RoboRelay Q-learning agent.")
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = RoboRelayEnv()
    env.reset(seed=args.seed)  # seeds the item positions for the whole training
    agent = QLearningAgent(seed=args.seed)

    record_every = args.episodes // 10  # 11 recorded games: 0, 10%, 20%, ..., and the last one

    # Settings saved next to the results, to know later how they were obtained
    config = {
        "episodes": args.episodes,
        "seed": args.seed,
        "learning_rate": agent.lr,
        "discount_factor": agent.gamma,
        "epsilon_decay": agent.epsilon_decay,
        "epsilon_min": agent.epsilon_min,
    }
    log = {"config": config, "reward": [], "success": [], "turns": [], "epsilon": []}
    history = []

    for episode in range(args.episodes):
        record = episode % record_every == 0 or episode == args.episodes - 1
        epsilon_used = agent.epsilon  # epsilon during this game (it decays after)

        total_reward, details = run_episode(env, agent, record=record)

        log["reward"].append(round(float(total_reward), 3))
        log["success"].append(env.world.is_success())
        log["turns"].append(env.world.turn_count)
        log["epsilon"].append(round(epsilon_used, 4))

        if record:
            details.update({
                "episode": episode,
                "epsilon": round(epsilon_used, 4),
                "total_reward": round(float(total_reward), 3),
                "turns": env.world.turn_count,
                "success": env.world.is_success(),
            })
            history.append(details)

        agent.decay_epsilon()

        if (episode + 1) % 500 == 0:
            last = log["reward"][-500:]
            success_rate = sum(log["success"][-500:]) / len(last) * 100
            print(f"Game {episode + 1}/{args.episodes} - mean reward (last 500): "
                  f"{sum(last) / len(last):.2f} - success: {success_rate:.0f}% - epsilon: {agent.epsilon:.3f}")

    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    print()
    print(f"Training log saved:   {save_json(log, 'training_log.json')}")
    print(f"Recorded games saved: {save_json(history, 'training_history.json')}")
    model_path = os.path.join(RECORDINGS_DIR, "q_agent.pkl")
    agent.save(model_path)
    print(f"Agent saved:          {model_path}")


if __name__ == "__main__":
    main()