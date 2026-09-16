from rl.gym_env import RoboRelayEnv
from rl.q_learning import QLearningAgent
import numpy as np

env = RoboRelayEnv()
agent = QLearningAgent()

# --- Ré-entraînement rapide (identique à train.py) ---
n_episodes = 20000
for episode in range(n_episodes):
    obs, info = env.reset(randomize=True)
    done = False
    while not done:
        carrier_action, placer_action = agent.choose_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(
            np.array([carrier_action, placer_action])
        )
        agent.update(obs, carrier_action, placer_action, reward, next_obs, terminated)
        obs = next_obs
        done = terminated or truncated
    agent.decay_epsilon()

print(f"Entraînement terminé. Q-table: {len(agent.q_table)} états appris.\n")

# --- Évaluation : plus aucune exploration ---
agent.epsilon = 0.0
n_eval_episodes = 100
successes = 0
turns_on_success = []

for episode in range(n_eval_episodes):
    obs, info = env.reset(randomize=True)
    done = False
    while not done:
        carrier_action, placer_action = agent.choose_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(
            np.array([carrier_action, placer_action])
        )
        obs = next_obs
        done = terminated or truncated

    if env.world.total_score() >= 1:
        successes += 1
        turns_on_success.append(env.world.turn_count)

print(f"Taux de réussite : {successes}/{n_eval_episodes} ({successes/n_eval_episodes*100:.0f}%)")
if turns_on_success:
    avg_turns = sum(turns_on_success) / len(turns_on_success)
    print(f"Nombre de tours moyen quand réussi : {avg_turns:.1f}")