import json
import os

import numpy as np

from rl.gym_env import RoboRelayEnv
from rl.q_learning import QLearningAgent

RECORDINGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "recordings")


def run_episode_with_recording(env, agent, episode_number):
    """
    Joue un épisode complet en laissant l'agent choisir ses actions
    normalement (avec son epsilon actuel), et enregistre tout ce qui
    se passe pour pouvoir l'inspecter/comparer plus tard.
    """
    obs, info = env.reset(randomize=True)

    record = {
        "episode": episode_number,
        "epsilon_at_start": agent.epsilon,
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
        next_obs, reward, terminated, truncated, info = env.step(
            np.array([carrier_action, placer_action])
        )
        agent.update(obs, carrier_action, placer_action, reward, next_obs, terminated)

        record["actions"].append([int(carrier_action), int(placer_action)])
        record["rewards"].append(round(float(reward), 3))
        total_reward += reward

        obs = next_obs
        done = terminated or truncated

    record["total_reward"] = round(total_reward, 3)
    record["turns"] = env.world.turn_count
    record["success"] = env.world.total_score() >= env.world.total_objects()

    return record


if __name__ == "__main__":
    env = RoboRelayEnv()
    agent = QLearningAgent()

    n_episodes = 5000
    checkpoint_every = n_episodes // 10  # 10 snapshots répartis sur l'entraînement

    episode_rewards = []
    history = []  # liste de tous les enregistrements d'épisodes-checkpoints

    for episode in range(n_episodes):
        is_checkpoint = (episode % checkpoint_every == 0) or (episode == n_episodes - 1)

        if is_checkpoint:
            record = run_episode_with_recording(env, agent, episode)
            total_reward = record["total_reward"]
            history.append(record)
        else:
            obs, info = env.reset(randomize=True)
            total_reward = 0
            done = False
            while not done:
                carrier_action, placer_action = agent.choose_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(
                    np.array([carrier_action, placer_action])
                )
                agent.update(obs, carrier_action, placer_action, reward, next_obs, terminated)
                total_reward += reward
                obs = next_obs
                done = terminated or truncated

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        if (episode + 1) % 500 == 0:
            avg_reward = sum(episode_rewards[-500:]) / 500
            print(f"Episode {episode+1}/{n_episodes} - reward moyen (500 derniers): {avg_reward:.2f} - epsilon: {agent.epsilon:.3f}")

    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    history_path = os.path.join(RECORDINGS_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nHistorique de {len(history)} épisodes sauvegardé : {history_path}")