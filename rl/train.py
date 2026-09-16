from rl.gym_env import RoboRelayEnv
from rl.q_learning import QLearningAgent
import numpy as np

env = RoboRelayEnv()
agent = QLearningAgent()

n_episodes = 5000
episode_rewards = []

for episode in range(n_episodes):
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

    # Affiche un point d'avancement de temps en temps
    if (episode + 1) % 500 == 0:
        avg_reward = sum(episode_rewards[-500:]) / 500
        print(f"Episode {episode+1}/{n_episodes} - reward moyen (500 derniers): {avg_reward:.2f} - epsilon: {agent.epsilon:.3f}")