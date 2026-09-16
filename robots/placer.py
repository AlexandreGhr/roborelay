from env.cell_types import CellType
from env.grid import Grid
from robots.base_robot import BaseRobot

DIRECTIONS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


class Placer(BaseRobot):
    def __init__(self, robot_id: str, x: int, y: int, grid: Grid):
        super().__init__(
            robot_id=robot_id,
            x=x,
            y=y,
            grid=grid,
            speed=0,          # ne bouge jamais
            vision_range=1,    # voit seulement ses cases adjacentes
        )
        self.storage: list[str] = []  # liste des objets stockés

    def charge(self, direction: str) -> bool:

        if direction not in DIRECTIONS:
            raise ValueError(f"Direction inconnue : {direction}")

        dx, dy = DIRECTIONS[direction]
        target_x, target_y = self.x + dx, self.y + dy

        if not self.grid.in_bounds(target_x, target_y):
            raise ValueError("Position hors grille")
        if self.grid.get(target_x, target_y) != CellType.OBJECT:
            raise ValueError("Aucun objet à charger sur cette case")

        # On retire l'objet de la grille, il "disparaît" dans le stockage
        self.grid.set(target_x, target_y, CellType.EMPTY)
        self.storage.append("object")  # pour l'instant, un objet générique
        return True

    @property
    def stored_count(self) -> int:
        return len(self.storage)

    def __repr__(self) -> str:
        return (
            f"Placer(id={self.robot_id}, pos=({self.x},{self.y}), "
            f"stored={self.stored_count})"
        )