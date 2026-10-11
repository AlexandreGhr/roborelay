"""
The game world: the grid, the robots, the items, the turn counter and the score.

The world can be reset to its starting setup, which training needs to do
thousands of times.
"""

import random

from env.cell_types import CellType
from env.grid import Grid
from robots.carrier import Carrier
from robots.placer import Placer
from robots.scout import Scout

# For each robot type: its class and the cell type it shows on the grid
ROBOT_TYPES = {
    "scout": (Scout, CellType.SCOUT),
    "carrier": (Carrier, CellType.CARRIER),
    "placer": (Placer, CellType.PLACER),
}


class World:
    def __init__(self, width: int, height: int, max_turns: int = 100, seed=None):
        self.width = width
        self.height = height
        self.max_turns = max_turns

        # Own random generator, so a seed gives the exact same games again
        self.rng = random.Random(seed)

        # Starting setup, remembered so reset() can rebuild the world
        self._spawn_config = []   # list of (robot_type, robot_id, x, y)
        self.objects_config = []  # starting item positions (x, y)

        self.grid = Grid(width, height)
        self.scouts = []
        self.carriers = []
        self.placers = []
        self.turn_count = 0
        self.current_object_positions = []  # item positions at the start of this game

    # --- Setup -----------------------------------------------------------

    def _create_robot(self, robot_type, robot_id, x, y):
        """Create a robot, put it on the grid and add it to the right list."""
        robot_class, cell_type = ROBOT_TYPES[robot_type]
        robot = robot_class(robot_id, x, y, self.grid)
        self.grid.set(x, y, cell_type)
        getattr(self, robot_type + "s").append(robot)  # self.scouts, self.carriers...
        return robot

    def add_scout(self, robot_id, x, y):
        self._spawn_config.append(("scout", robot_id, x, y))
        return self._create_robot("scout", robot_id, x, y)

    def add_carrier(self, robot_id, x, y):
        self._spawn_config.append(("carrier", robot_id, x, y))
        return self._create_robot("carrier", robot_id, x, y)

    def add_placer(self, robot_id, x, y):
        self._spawn_config.append(("placer", robot_id, x, y))
        return self._create_robot("placer", robot_id, x, y)

    def add_object(self, x, y):
        if not self.grid.is_empty(x, y):
            raise ValueError(f"Cell ({x}, {y}) is not empty, cannot place an item")
        self.grid.set(x, y, CellType.OBJECT)
        self.objects_config.append((x, y))
        self.current_object_positions = self.grid.positions_of(CellType.OBJECT)

    # --- Reset -----------------------------------------------------------

    def reset(self, randomize: bool = False):
        """Rebuild the world from its starting setup.

        randomize=False: items go back to their starting positions.
        randomize=True: same number of items, at random empty cells.
        """
        # Robots are rebuilt as NEW objects: any old reference to a robot
        # (for example in the gym env) must be updated after a reset.
        self.grid = Grid(self.width, self.height)
        self.scouts = []
        self.carriers = []
        self.placers = []
        self.turn_count = 0

        for robot_type, robot_id, x, y in self._spawn_config:
            self._create_robot(robot_type, robot_id, x, y)

        if randomize:
            empty_cells = self.grid.positions_of(CellType.EMPTY)
            count = min(len(self.objects_config), len(empty_cells))
            item_positions = self.rng.sample(empty_cells, k=count)
        else:
            item_positions = self.objects_config

        for x, y in item_positions:
            self.grid.set(x, y, CellType.OBJECT)

        self.current_object_positions = self.grid.positions_of(CellType.OBJECT)

    # --- Game state ------------------------------------------------------

    def increment_turn(self):
        self.turn_count += 1

    def total_score(self):
        """Number of items stored by all Placers."""
        return sum(placer.stored_count for placer in self.placers)

    def total_objects(self):
        return len(self.objects_config)

    def is_success(self):
        """True when every item has been stored."""
        return self.total_score() >= self.total_objects()

    def is_done(self):
        """The game ends on success or when the turn limit is reached."""
        return self.is_success() or self.turn_count >= self.max_turns