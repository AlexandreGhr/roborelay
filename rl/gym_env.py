import gymnasium as gym 
import numpy as np
from gymnasium import spaces
from env.world import World
from env.cell_types import CellType
from robots.carrier import DIRECTIONS

class RoboRelayEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.world = World(5, 5, max_turns=50)
        self.carrier = self.world.add_carrier("carrier-1", 0, 0)
        self.placer = self.world.add_placer("placer-1", 4, 4)
        self.world.add_object(2, 2)

        self.action_space = spaces.MultiDiscrete([6, 2])
        self.observation_space = spaces.Box(low=0, high=5, shape=(5,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.world.reset()
        self.carrier = self.world.carriers[0]
        self.placer = self.world.placers[0]

        observation = self._get_observation()
        info = {}
        return observation, info

    def _get_observation(self):
        if self.carrier.carrying_object:
            target_x, target_y = self.placer.x, self.placer.y
        else:
            target_x, target_y = self._find_object_position()

        return np.array([
            self.carrier.x,
            self.carrier.y,
            target_x,
            target_y,
            1.0 if self.carrier.carrying_object else 0.0,
        ], dtype=np.float32)

    def _find_object_position(self):
        for y in range(self.world.grid.height):
            for x in range(self.world.grid.width):
                if self.world.grid.get(x, y) == CellType.OBJECT:
                    return (x, y)
        return (-1, -1)  # aucun objet trouvé sur la grille

    
    def step(self, action):
        carrier_action, placer_action = action
        reward = 0.0

        # --- Actions du Carrier ---
        if carrier_action == 0:
            self.carrier.facing = "up"
            self.carrier.move("up")
        elif carrier_action == 1:
            self.carrier.facing = "down"
            self.carrier.move("down")
        elif carrier_action == 2:
            self.carrier.facing = "left"
            self.carrier.move("left")
        elif carrier_action == 3:
            self.carrier.facing = "right"
            self.carrier.move("right")

        elif carrier_action == 4:  # pick_up
            try:
                self.carrier.pick_up(self.carrier.facing)
                reward += 1.0
            except ValueError:
                reward -= 0.1

        elif carrier_action == 5:  # drop
            dx, dy = DIRECTIONS[self.carrier.facing]
            drop_x, drop_y = self.carrier.x + dx, self.carrier.y + dy
            try:
                self.carrier.drop(self.carrier.facing)
                if self._is_adjacent_to_placer(drop_x, drop_y):
                    reward += 3.0   # bon geste : posé près du Placer
                else:
                    reward -= 0.5   # posé n'importe où, pas utile
            except ValueError:
                reward -= 0.1

        # --- Action du Placer ---
        if placer_action == 1:  # tente de charger
            charged = False
            for direction in ["up", "down", "left", "right"]:
                try:
                    self.placer.charge(direction)
                    reward += 5.0
                    charged = True
                    break
                except ValueError:
                    continue
            if not charged:
                reward -= 0.5
        # placer_action == 0 -> ne rien faire, reward inchangé (neutre)

        self.world.increment_turn()
        reward -= 0.01

        terminated = self.world.is_done()
        truncated = False
        observation = self._get_observation()
        info = {}

        return observation, reward, terminated, truncated, info

    def _is_adjacent_to_placer(self, x, y):
        for placer in self.world.placers:
            if abs(placer.x - x) + abs(placer.y - y) == 1:
                return True
        return False