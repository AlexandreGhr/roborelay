from env.grid import Grid
from env.cell_types import CellType
from robots.scout import Scout
from robots.carrier import Carrier
from robots.placer import Placer
import random

class World():
    def __init__(self, width: int, height: int, max_turns: int = 100):
        self.grid = Grid(width, height)
        self.scouts = []
        self.carriers = []
        self.placers = []

        self.width = width
        self.height = height
        self._spawn_config = []  # liste de tuples (type_robot, robot_id, x, y)
        self.objects_config = []  # positions des objets fixes, pour le mode non-random

        self.max_turns = max_turns
        self.turn_count = 0


    def add_scout(self, robot_id, x, y):
        scout = Scout(robot_id, x, y, self.grid)
        self.grid.set(x, y, CellType.SCOUT)
        self.scouts.append(scout)
        self._spawn_config.append(("scout", robot_id, x, y))
        return scout

    def add_carrier(self, robot_id, x, y):
        carrier = Carrier(robot_id, x, y, self.grid)
        self.grid.set(x, y, CellType.CARRIER)
        self.carriers.append(carrier)
        self._spawn_config.append(("carrier", robot_id, x, y))
        return carrier

    def add_placer(self, robot_id, x, y):
        placer = Placer(robot_id, x, y, self.grid)
        self.grid.set(x, y, CellType.PLACER)
        self.placers.append(placer)
        self._spawn_config.append(("placer", robot_id, x, y))
        return placer

    def total_score(self):
        total_score = 0
        for placer in self.placers:
            total_score += placer.stored_count
        return total_score

    def add_object(self, x, y):
        if not self.grid.is_empty(x, y):
            raise ValueError(f"Case ({x}, {y}) déjà occupée, impossible d'y placer un objet")
        self.grid.set(x, y, CellType.OBJECT)
        self.objects_config.append((x, y))

    def reset(self, randomize: bool = False):
        # 1. Grille toute neuve
        self.grid = Grid(self.width, self.height)

        # 2. Vider les listes de robots
        self.scouts = []
        self.carriers = []
        self.placers = []

        self.turn_count = 0

        # 3. Recréer chaque robot à sa position de spawn mémorisée.
        for robot_type, robot_id, x, y in self._spawn_config:
            if robot_type == "scout":
                scout = Scout(robot_id, x, y, self.grid)
                self.grid.set(x, y, CellType.SCOUT)
                self.scouts.append(scout)
            elif robot_type == "carrier":
                carrier = Carrier(robot_id, x, y, self.grid)
                self.grid.set(x, y, CellType.CARRIER)
                self.carriers.append(carrier)
            elif robot_type == "placer":
                placer = Placer(robot_id, x, y, self.grid)
                self.grid.set(x, y, CellType.PLACER)
                self.placers.append(placer)

        # 4. Replacer les objets
        if not randomize:
            # Mode fixe : exactement les mêmes positions qu'au départ
            for x, y in self.objects_config:
                self.grid.set(x, y, CellType.OBJECT)
        else:
            # Mode aléatoire : même NOMBRE d'objets, positions aléatoires
            num_objects = len(self.objects_config)
            empty_cells = [
                (x, y)
                for x in range(self.width)
                for y in range(self.height)
                if self.grid.is_empty(x, y)
            ]
            chosen = random.sample(empty_cells, k=min(num_objects, len(empty_cells)))
            for x, y in chosen:
                self.grid.set(x, y, CellType.OBJECT)

        self.current_object_positions = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if self.grid.get(x, y) == CellType.OBJECT
        ]

    def increment_turn(self):
        self.turn_count += 1

    def total_objects(self):
        return len(self.objects_config)

    def is_done(self):
        if self.total_score() >= self.total_objects() or self.turn_count >= self.max_turns : 
            return True
        return False
        

    