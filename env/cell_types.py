"""
Types of content a grid cell can hold.

Rule: a cell holds at most ONE thing at a time.
"""

from enum import Enum


class CellType(Enum):
    EMPTY = "."      # Nothing in the cell
    OBJECT = "O"     # Item to pick up and store
    SCOUT = "S"      # Scout robot (explores, sees far)
    CARRIER = "C"    # Carrier robot (moves and carries items)
    PLACER = "P"     # Placer robot (fixed, stores items)