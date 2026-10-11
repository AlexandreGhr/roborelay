"""
Measure what the trained agent really learned: no random actions, no learning.

    python3 -m rl.evaluate                       # 500 games, seed 1000
    python3 -m rl.evaluate --episodes 1000 --seed 7

The evaluation seed is different from the training seed (0): like a test set,
the agent is measured on games it did not train on, and always on the same ones.
Saves the results in recordings/evaluation.json.
"""

import argparse
import json
import os

import numpy as np

from rl.gym_env import GRID_SIZE, RoboRelayEnv
from rl.q_learning import QLearningAgent

RECORDINGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "recordings")


def manhattan(a, b):
    """Number of steps between two cells on a grid without diagonals."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def neighbours(pos):
    """The 4 cells touching pos by one side, inside the grid."""
    x, y = pos
    candidates = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]
    return [(nx, ny) for nx, ny in candidates if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE]


def turns_lower_bound(carrier_start, placer_pos, item_pos):
    """Fewest turns the game could take. The agent can never do better.

    Plan: walk next to the item, pick it up (1 turn), walk to a cell from where
    the item can be dropped next to the Placer, drop it while the Placer charges
    in the same turn (1 turn).

    It is a LOWER BOUND, not always reachable: it ignores obstacles on the way,
    and the extra turns sometimes needed to face the right direction
    (the Carrier can only turn by trying to move).
    """
    placer_pos = tuple(placer_pos)
    if manhattan(item_pos, placer_pos) == 1:
        return 1  # the item already touches the Placer: it only has to charge

    # Cells where the Carrier can stand to pick up the item (never the Placer's cell)
    pickup_cells = [c for c in neighbours(item_pos) if c != placer_pos]

    # Cells where the Carrier can stand to drop the item next to the Placer
    drop_cells = {s for d in neighbours(placer_pos) for s in neighbours(d) if s != placer_pos}

    best = None
    for pickup in pickup_cells:
        to_item = manhattan(carrier_start, pickup)
        for stand in drop_cells:
            total = to_item + 1 + manhattan(pickup, stand) + 1
            if best is None or total < best:
                best = total
    return best


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained RoboRelay agent.")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--agent", default=os.path.join(RECORDINGS_DIR, "q_agent.pkl"))
    args = parser.parse_args()

    env = RoboRelayEnv()
    env.reset(seed=args.seed)
    agent = QLearningAgent.load(args.agent, seed=args.seed)  # epsilon = 0: no random actions
    print(f"Agent loaded: {len(agent.q_table)} states in the Q-table.\n")

    successes = 0
    turns = []
    ratios = []  # turns played / lower bound, for successful games

    for _ in range(args.episodes):
        obs, _ = env.reset(randomize=True)
        carrier_start = env.carrier.position
        placer_pos = env.placer.position
        item_pos = env.world.current_object_positions[0]

        done = False
        while not done:
            carrier_action, placer_action = agent.choose_action(obs)
            obs, _, terminated, truncated, _ = env.step(np.array([carrier_action, placer_action]))
            done = terminated or truncated

        if env.world.is_success():
            successes += 1
            turns.append(env.world.turn_count)
            ratios.append(env.world.turn_count / turns_lower_bound(carrier_start, placer_pos, item_pos))

    results = {
        "agent": os.path.basename(args.agent),
        "episodes": args.episodes,
        "seed": args.seed,
        "success_rate": successes / args.episodes,
        "mean_turns": float(np.mean(turns)) if turns else None,
        "mean_ratio_to_lower_bound": float(np.mean(ratios)) if ratios else None,
        "games_at_lower_bound": sum(1 for r in ratios if r <= 1.0),
        "worst_ratio": max(ratios) if ratios else None,
    }

    print(f"Success rate: {successes}/{args.episodes} ({results['success_rate'] * 100:.1f}%)")
    if ratios:
        print(f"Mean turns per game: {results['mean_turns']:.2f}")
        print(f"Turns compared to the lower bound: {results['mean_ratio_to_lower_bound']:.2f}x on average")
        print(f"Games that reach the lower bound (surely optimal): "
              f"{results['games_at_lower_bound']}/{len(ratios)}")
        print(f"Worst successful game: {results['worst_ratio']:.2f}x")

    out_path = os.path.join(RECORDINGS_DIR, "evaluation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()