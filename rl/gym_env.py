"""
Gymnasium environment: turns the RoboRelay game into a reinforcement learning problem.

One agent controls two robots at the same time:
- the Carrier (6 actions): move up / down / left / right, pick up, drop
- the Placer  (2 actions): wait, charge

Goal: bring the item next to the Placer, so the Placer can store it.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.cell_types import CellType
from env.world import World

GRID_SIZE = 5
MAX_TURNS = 50

CARRIER_MOVES = ["up", "down", "left", "right"]  # carrier actions 0 to 3
PICK_UP, DROP = 4, 5                             # carrier actions 4 and 5
CHARGE = 1                                       # placer action 1 (0 = wait)

FACING_TO_INDEX = {"up": 0, "down": 1, "left": 2, "right": 3}

# --- Reward design ---------------------------------------------------------
# The big reward is for the real goal (storing the item). Smaller bonuses
# guide the agent on the way, and are paid only ONCE per game: otherwise the
# agent learns to drop and pick up the item again and again to farm points.
REWARD_PICK_UP = 1.0        # first pick up of the game
REWARD_DROP_NEAR = 3.0      # first drop next to the Placer
REWARD_STORE = 5.0          # the Placer stores the item (the real goal)
PENALTY_DROP_FAR = -0.5     # drop somewhere useless
PENALTY_FAILED_CHARGE = -0.5  # the Placer tries to charge with nothing next to it
PENALTY_INVALID = -0.1      # pick up / drop that fails
PENALTY_PER_TURN = -0.01    # small cost every turn, so faster is better


class RoboRelayEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.world = World(GRID_SIZE, GRID_SIZE, max_turns=MAX_TURNS)
        self.carrier = self.world.add_carrier("carrier-1", 0, 0)
        self.placer = self.world.add_placer("placer-1", GRID_SIZE - 1, GRID_SIZE - 1)
        self.world.add_object(2, 2)

        # Each turn the agent picks one action per robot: [carrier_action, placer_action]
        self.action_space = spaces.MultiDiscrete([6, 2])

        # What the agent sees (6 numbers):
        # [carrier_x, carrier_y, target_x, target_y, carrying (0/1), facing (0-3)]
        # target = the item, or the Placer once the item is carried.
        # target can be -1 when there is no item left on the grid.
        self.observation_space = spaces.Box(
            low=-1, high=GRID_SIZE - 1, shape=(6,), dtype=np.float32
        )

    def reset(self, seed=None, options=None, randomize=False):
        super().reset(seed=seed)
        if seed is not None:
            # The world has its own random generator (item positions): seed it too,
            # so the same seed always gives the same games.
            self.world.rng.seed(seed)
        self.world.reset(randomize=randomize)

        # world.reset() creates NEW robot objects: update our references to them
        self.carrier = self.world.carriers[0]
        self.placer = self.world.placers[0]

        # The small bonuses can be earned once per game
        self.pickup_rewarded = False
        self.drop_rewarded = False

        return self._get_observation(), {}

    def step(self, action):
        carrier_action, placer_action = action
        reward = PENALTY_PER_TURN

        # --- Carrier ---
        if carrier_action < len(CARRIER_MOVES):
            # One action turns AND moves the Carrier (in the keyboard game, these are two key presses)
            direction = CARRIER_MOVES[carrier_action]
            self.carrier.facing = direction
            self.carrier.move(direction)  # returns False against a wall or a robot: no penalty

        elif carrier_action == PICK_UP:
            try:
                self.carrier.pick_up(self.carrier.facing)
                if not self.pickup_rewarded:
                    reward += REWARD_PICK_UP
                    self.pickup_rewarded = True
            except ValueError:
                reward += PENALTY_INVALID

        elif carrier_action == DROP:
            drop_x, drop_y = self.carrier.adjacent(self.carrier.facing)
            try:
                self.carrier.drop(self.carrier.facing)
                if self._is_next_to_placer(drop_x, drop_y):
                    if not self.drop_rewarded:
                        reward += REWARD_DROP_NEAR
                        self.drop_rewarded = True
                else:
                    reward += PENALTY_DROP_FAR
            except ValueError:
                reward += PENALTY_INVALID

        # --- Placer (acts after the Carrier: it can store an item dropped this same turn) ---
        if placer_action == CHARGE:
            if self._placer_charge_any():
                reward += REWARD_STORE
            else:
                reward += PENALTY_FAILED_CHARGE

        self.world.increment_turn()

        # terminated: the game itself is over (item stored).
        # truncated: the game is cut by the turn limit. The state is not a real end:
        # the agent must not learn "nothing more can be won from here".
        terminated = self.world.is_success()
        truncated = not terminated and self.world.turn_count >= self.world.max_turns
        info = {"is_success": terminated}

        return self._get_observation(), reward, terminated, truncated, info

    # --- Helpers -------------------------------------------------------------

    def _get_observation(self):
        if self.carrier.carrying_object:
            target_x, target_y = self.placer.position
        else:
            target_x, target_y = self._item_position()

        return np.array([
            self.carrier.x,
            self.carrier.y,
            target_x,
            target_y,
            1.0 if self.carrier.carrying_object else 0.0,
            FACING_TO_INDEX[self.carrier.facing],
        ], dtype=np.float32)

    def _item_position(self):
        """Position of the item on the grid, or (-1, -1) if there is none."""
        positions = self.world.grid.positions_of(CellType.OBJECT)
        return positions[0] if positions else (-1, -1)

    def _placer_charge_any(self):
        """The Placer stores an item from any of its 4 neighbour cells. Return True if it did."""
        for direction in ["up", "down", "left", "right"]:
            try:
                self.placer.charge(direction)
                return True
            except ValueError:
                continue
        return False

    def _is_next_to_placer(self, x, y):
        """True if (x, y) touches a Placer by one side (no diagonals)."""
        return any(abs(p.x - x) + abs(p.y - y) == 1 for p in self.world.placers)