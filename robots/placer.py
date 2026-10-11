"""
Placer robot: fixed, it never moves (it has no move() method).
It takes an item dropped on a neighbour cell and stores it.
The number of stored items is the score of the game.
"""

from env.cell_types import CellType
from robots.base_robot import BaseRobot


class Placer(BaseRobot):
    CELL_TYPE = CellType.PLACER

    def __init__(self, robot_id: str, x: int, y: int, grid):
        super().__init__(robot_id, x, y, grid, vision_range=1)
        # A list (not just a counter) so items can carry more info later (type, weight...)
        self.storage: list[str] = []

    def charge(self, direction: str) -> bool:
        """Take the item from the neighbour cell in this direction and store it."""
        target_x, target_y = self.adjacent(direction)

        if not self.grid.in_bounds(target_x, target_y):
            raise ValueError("Position is outside the grid")
        # Check for an ITEM, not just "not empty": otherwise the Placer
        # could store another robot.
        if self.grid.get(target_x, target_y) != CellType.OBJECT:
            raise ValueError("No item to charge in this cell")

        self.grid.set(target_x, target_y, CellType.EMPTY)  # the item leaves the grid...
        self.storage.append("item")                         # ...and goes into the storage
        return True

    @property
    def stored_count(self) -> int:
        return len(self.storage)

    def __repr__(self) -> str:
        return (f"Placer(id={self.robot_id}, pos=({self.x},{self.y}), "
                f"stored={self.stored_count})")