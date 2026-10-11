"""
Base classes shared by all robots.

BaseRobot     -> what every robot has (id, position, grid, facing)
MobileRobot   -> a robot that can move (Scout, Carrier)
The Placer inherits BaseRobot directly: it has no move() method at all.
"""

from env.cell_types import CellType
from env.grid import Grid

# Direction name -> (dx, dy). y grows downwards, so "up" is -1.
DIRECTIONS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


class BaseRobot:
    CELL_TYPE = None  # set by each robot class, e.g. CellType.CARRIER

    def __init__(self, robot_id: str, x: int, y: int, grid: Grid, vision_range: int = 1):
        self.robot_id = robot_id
        self.x = x
        self.y = y
        self.grid = grid
        self.facing = "down"  # direction the robot looks at; pick_up/drop/charge act this way

        # Not used yet: planned for local vision (each robot only sees nearby cells)
        self.vision_range = vision_range

    @property
    def position(self) -> tuple[int, int]:
        return (self.x, self.y)

    def adjacent(self, direction: str) -> tuple[int, int]:
        """Return the position of the neighbour cell in this direction."""
        if direction not in DIRECTIONS:
            raise ValueError(f"Unknown direction: {direction}")
        dx, dy = DIRECTIONS[direction]
        return (self.x + dx, self.y + dy)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.robot_id}, pos=({self.x},{self.y}))"


class MobileRobot(BaseRobot):
    def move(self, direction: str) -> bool:
        """Move one cell. Return False if the cell is outside the grid or not empty."""
        new_x, new_y = self.adjacent(direction)

        if not self.grid.in_bounds(new_x, new_y):
            return False
        if not self.grid.is_empty(new_x, new_y):
            return False

        self.grid.set(self.x, self.y, CellType.EMPTY)
        self.grid.set(new_x, new_y, self.CELL_TYPE)
        self.x, self.y = new_x, new_y
        return True