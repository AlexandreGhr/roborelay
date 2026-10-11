"""
Carrier robot: moves around, picks up an item and drops it next to the Placer.
It can only act on a neighbour cell (up, down, left or right), never on its own cell.
"""

from env.cell_types import CellType
from robots.base_robot import MobileRobot


class Carrier(MobileRobot):
    CELL_TYPE = CellType.CARRIER

    def __init__(self, robot_id: str, x: int, y: int, grid):
        super().__init__(robot_id, x, y, grid, vision_range=1)
        self.carrying_object = False  # only the Carrier can carry an item

    def pick_up(self, direction: str) -> bool:
        """Take the item from the neighbour cell in this direction."""
        if self.carrying_object:
            raise ValueError("Already carrying an item")

        target_x, target_y = self.adjacent(direction)
        if not self.grid.in_bounds(target_x, target_y):
            raise ValueError("Position is outside the grid")
        # Check for an ITEM, not just "not empty": otherwise the Carrier
        # could pick up another robot.
        if self.grid.get(target_x, target_y) != CellType.OBJECT:
            raise ValueError("No item to pick up in this cell")

        self.grid.set(target_x, target_y, CellType.EMPTY)
        self.carrying_object = True
        return True

    def drop(self, direction: str) -> bool:
        """Put the carried item on the empty neighbour cell in this direction."""
        if not self.carrying_object:
            raise ValueError("No item to drop")

        target_x, target_y = self.adjacent(direction)
        if not self.grid.in_bounds(target_x, target_y):
            raise ValueError("Position is outside the grid")
        if not self.grid.is_empty(target_x, target_y):
            raise ValueError("Cell is not empty, cannot drop here")

        self.grid.set(target_x, target_y, CellType.OBJECT)
        self.carrying_object = False
        return True

    def __repr__(self) -> str:
        return (f"Carrier(id={self.robot_id}, pos=({self.x},{self.y}), "
                f"carrying={self.carrying_object})")