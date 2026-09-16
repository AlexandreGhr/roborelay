from rl.gym_env import RoboRelayEnv
import numpy as np

print("=== Test 1 : cycle complet (drop près du Placer + charge réussi) ===")
env = RoboRelayEnv()
obs, info = env.reset()

sequence_1 = [
    ([3, 0], 'right'),
    ([3, 0], 'right'),
    ([1, 0], 'down'),
    ([4, 0], 'pick_up'),
    ([3, 0], 'right'),
    ([1, 0], 'down'),
    ([1, 0], 'down'),
    ([5, 0], 'drop'),       # attendu : +2.99 (posé adjacent au Placer)
    ([0, 1], 'placer charge'),  # attendu : +4.99, score=1, terminated=True
]
for act, label in sequence_1:
    obs, reward, terminated, truncated, info = env.step(np.array(act))
    print(f'{label:15} reward={reward:.2f} score={env.world.total_score()} terminated={terminated}')

print()
print("=== Test 2 : drop valide mais loin du Placer ===")
env2 = RoboRelayEnv()
obs, info = env2.reset()

sequence_2 = [
    ([3, 0], 'right'),
    ([3, 0], 'right'),
    ([1, 0], 'down'),
    ([4, 0], 'pick_up'),
    ([3, 0], 'right'),
    ([5, 0], 'drop'),       # attendu : -0.51 (posé loin du Placer)
]
for act, label in sequence_2:
    obs, reward, terminated, truncated, info = env2.step(np.array(act))
    print(f'{label:10} reward={reward:.2f} pos={env2.carrier.position} carrying={env2.carrier.carrying_object}')

print()
print("=== Test 3 : Placer tente de charger alors que rien n'est disponible ===")
env3 = RoboRelayEnv()
obs, info = env3.reset()
obs, reward, terminated, truncated, info = env3.step(np.array([0, 1]))
print(f'placer charge (rien dispo) reward={reward:.2f} score={env3.world.total_score()}')