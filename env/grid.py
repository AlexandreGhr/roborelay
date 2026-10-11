"""
The game board: a 2D grid of cells.

Coordinates are (x, y): x = column, y = row.
The origin (0, 0) is the top-left corner.
"""

from env.cell_types import CellType


class Grid:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        # Stored as a list of rows, so a cell is accessed with _cells[y][x].
        # Use get() and set() instead of touching _cells directly.
        self._cells = [
            [CellType.EMPTY for _ in range(width)]
            for _ in range(height)
        ]

    def in_bounds(self, x: int, y: int) -> bool:
        """Return True if (x, y) is inside the grid."""
        return 0 <= x < self.width and 0 <= y < self.height

    def get(self, x: int, y: int) -> CellType:
        """Return what is in the cell (x, y)."""
        if not self.in_bounds(x, y):
            raise ValueError(f"Position ({x}, {y}) is outside the grid")
        return self._cells[y][x]

    def set(self, x: int, y: int, cell_type: CellType) -> None:
        """Put cell_type in the cell (x, y)."""
        if not self.in_bounds(x, y):
            raise ValueError(f"Position ({x}, {y}) is outside the grid")
        self._cells[y][x] = cell_type

    def is_empty(self, x: int, y: int) -> bool:
        """Return True if the cell (x, y) holds nothing."""
        return self.get(x, y) == CellType.EMPTY

    def positions_of(self, cell_type: CellType) -> list[tuple[int, int]]:
        """Return every position (x, y) that holds this cell type."""
        return [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if self._cells[y][x] == cell_type
        ]

    def __repr__(self) -> str:
        """Text view of the grid, one line per row (useful for debugging)."""
        rows = []
        for row in self._cells:
            rows.append(" ".join(cell.value for cell in row))
        return "\n".join(rows)