from env.cell_types import CellType
from env.grid import Grid
from robots.base_robot import BaseRobot

# Déplacements possibles : (dx, dy)
DIRECTIONS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


class Scout(BaseRobot):
    def __init__(self, robot_id: str, x: int, y: int, grid: Grid):
        super().__init__(
            robot_id=robot_id,
            x=x,
            y=y,
            grid=grid,
            speed=2,          # plus rapide que les autres robots
            vision_range=4,    # voit plus loin que les autres
        )

    def move(self, direction: str) -> bool:

        if direction not in DIRECTIONS:
            raise ValueError(f"Direction inconnue : {direction}")

        dx, dy = DIRECTIONS[direction]
        new_x, new_y = self.x + dx, self.y + dy

        if not self.grid.in_bounds(new_x, new_y):
            return False
        if not self.grid.is_empty(new_x, new_y):
            return False

        # Libère l'ancienne case, occupe la nouvelle
        self.grid.set(self.x, self.y, CellType.EMPTY)
        self.grid.set(new_x, new_y, CellType.SCOUT)
        self.x, self.y = new_x, new_y
        return True