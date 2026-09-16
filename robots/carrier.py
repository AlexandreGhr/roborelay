from env.cell_types import CellType
from env.grid import Grid
from robots.base_robot import BaseRobot

DIRECTIONS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


class Carrier(BaseRobot):
    def __init__(self, robot_id: str, x: int, y: int, grid: Grid):
        super().__init__(
            robot_id=robot_id,
            x=x,
            y=y,
            grid=grid,
            speed=1,
            vision_range=1,
        )
        # Pas besoin de redéfinir carrying_object ici,
        # BaseRobot.__init__() le fait déjà (self.carrying_object = False)

    def move(self, direction: str) -> bool:
        if direction not in DIRECTIONS:
            raise ValueError(f"Direction inconnue : {direction}")

        dx, dy = DIRECTIONS[direction]
        new_x, new_y = self.x + dx, self.y + dy

        if not self.grid.in_bounds(new_x, new_y):
            return False
        if not self.grid.is_empty(new_x, new_y):
            return False

        self.grid.set(self.x, self.y, CellType.EMPTY)
        self.grid.set(new_x, new_y, CellType.CARRIER)
        self.x, self.y = new_x, new_y
        return True

    def pick_up(self, direction: str) -> bool:
        if direction not in DIRECTIONS:
            raise ValueError(f"Direction inconnue : {direction}")

        dx, dy = DIRECTIONS[direction]
        new_x, new_y = self.x + dx, self.y + dy

        if not self.grid.in_bounds(new_x, new_y):
            raise ValueError("Position hors grille")
        if self.grid.is_empty(new_x, new_y):
            raise ValueError("Aucun objet à ramasser sur cette case")
        if self.carrying_object:
            raise ValueError("Robot déjà chargé")

        self.carrying_object = True
        self.grid.set(new_x, new_y, CellType.EMPTY)
        return True

    def drop(self, direction: str) -> bool:
        if not self.carrying_object:
            raise ValueError("Aucun objet à déposer")
        if direction not in DIRECTIONS:
            raise ValueError(f"Direction inconnue : {direction}")

        dx, dy = DIRECTIONS[direction]
        new_x, new_y = self.x + dx, self.y + dy

        if not self.grid.in_bounds(new_x, new_y):
            raise ValueError("Position hors grille")
        if not self.grid.is_empty(new_x, new_y):
            raise ValueError("Case déjà occupée, impossible de déposer ici")

        self.grid.set(new_x, new_y, CellType.OBJECT)
        self.carrying_object = False
        return True