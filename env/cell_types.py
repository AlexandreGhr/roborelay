from enum import Enum


class CellType(Enum):
    EMPTY = "."      # Case vide
    OBJECT = "O"      # Objet à récupérer/ranger
    SCOUT = "S"       # Robot Éclaireur
    CARRIER = "C"     # Robot Porteur
    PLACER = "P"      # Robot Placeur (fixe, point de dépôt)