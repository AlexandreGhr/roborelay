"""
Scout robot: moves around and is meant to see far.

For now it only moves. Its planned role is to explore with a wide vision
and tell the other robots where the items are.
"""

from env.cell_types import CellType
from robots.base_robot import MobileRobot


class Scout(MobileRobot):
    CELL_TYPE = CellType.SCOUT

    def __init__(self, robot_id: str, x: int, y: int, grid):
        super().__init__(robot_id, x, y, grid, vision_range=4)  # sees further than the others