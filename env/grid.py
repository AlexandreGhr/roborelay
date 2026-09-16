from env.cell_types import CellType


class Grid:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self._cells = [
            [CellType.EMPTY for _ in range(width)]
            for _ in range(height)
        ]

    def in_bounds(self, x: int, y: int) -> bool:
        """Vérifie que la position (x, y) est bien dans la grille."""
        return 0 <= x < self.width and 0 <= y < self.height

    def get(self, x: int, y: int) -> CellType:
        if not self.in_bounds(x, y):
            raise ValueError(f"Position ({x}, {y}) hors de la grille")
        return self._cells[y][x]

    def set(self, x: int, y: int, cell_type: CellType) -> None:
        if not self.in_bounds(x, y):
            raise ValueError(f"Position ({x}, {y}) hors de la grille")
        self._cells[y][x] = cell_type

    def is_empty(self, x: int, y: int) -> bool:
        return self.get(x, y) == CellType.EMPTY

    def __repr__(self) -> str:
        rows = []
        for row in self._cells:
            rows.append(" ".join(cell.value for cell in row))
        return "\n".join(rows)