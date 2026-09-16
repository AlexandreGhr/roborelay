from rl.gym_env import RoboRelayEnv
from rl.q_learning import QLearningAgent
import numpy as np

env = RoboRelayEnv()
agent = QLearningAgent()

obs, info = env.reset()
total_reward = 0

for step in range(20):
    carrier_action, placer_action = agent.choose_action(obs)
    next_obs, reward, terminated, truncated, info = env.step(np.array([carrier_action, placer_action]))

    agent.update(obs, carrier_action, placer_action, reward, next_obs, terminated)

    total_reward += reward
    obs = next_obs

    if terminated:
        print(f'Episode terminé au step {step}, score final: {env.world.total_score()}')
        break

print('Total reward:', round(total_reward, 2))
print('Taille de la Q-table (nb états visités):', len(agent.q_table))
print('Epsilon actuel:', agent.epsilon)
agent.decay_epsilon()
print('Epsilon après decay:', agent.epsilon)