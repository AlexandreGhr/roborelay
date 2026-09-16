from env.grid import Grid


class BaseRobot:
    def __init__(
        self,
        robot_id: str,
        x: int,
        y: int,
        grid: Grid,
        speed: int = 1,
        vision_range: int = 1,
        size: int = 1,
    ):
        self.robot_id = robot_id
        self.x = x
        self.y = y
        self.grid = grid

        self.speed = speed
        self.vision_range = vision_range
        self.size = size

        self.carrying_object = False

        self.facing = "down"  # direction vers laquelle le robot "regarde"

    @property
    def position(self) -> tuple[int, int]:
        return (self.x, self.y)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"id={self.robot_id}, pos=({self.x},{self.y}), "
            f"carrying={self.carrying_object})"
        )