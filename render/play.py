"""
Play RoboRelay yourself with the keyboard (no AI involved).

    python3 -m render.play

Controls:
    Tab      switch the active robot (black frame)
    Arrows   first press turns the robot, a second press in the same direction moves it
    Space    action: Carrier picks up / drops, Placer charges
    R        restart the game

Note: the keyboard rules differ a little from the agent's rules. Here, turning and
moving are two separate key presses (each one costs a turn). For the agent, one
"up" action turns AND moves in a single turn.
"""

import pygame

from env.world import World
from render.drawing import CELL_SIZE, LEGEND_WIDTH, draw_grid, draw_legend, draw_robot_details
from robots.base_robot import MobileRobot
from robots.carrier import Carrier
from robots.placer import Placer

HUD_HEIGHT = 90  # info panel above the grid (3 lines of text)

CONTROLS = [
    ("Tab", "Switch robot"),
    ("Arrow", "Turn (press again: move)"),
    ("Space", "Pick up / drop / charge"),
    ("R", "Restart"),
]

ARROW_TO_DIRECTION = {
    pygame.K_UP: "up",
    pygame.K_DOWN: "down",
    pygame.K_LEFT: "left",
    pygame.K_RIGHT: "right",
}


def create_world():
    """The starting setup of the keyboard game."""
    world = World(10, 8, max_turns=100)
    world.add_scout("scout-1", 0, 0)
    world.add_carrier("carrier-1", 2, 2)
    world.add_placer("placer-1", 8, 6)
    world.add_object(4, 3)
    world.add_object(6, 5)
    return world


def try_action(robot):
    """Space key: the action depends on the robot type. Invalid actions are just ignored."""
    try:
        if isinstance(robot, Carrier):
            if robot.carrying_object:
                robot.drop(robot.facing)
            else:
                robot.pick_up(robot.facing)
        elif isinstance(robot, Placer):
            robot.charge(robot.facing)
    except ValueError as error:
        print(f"Action not possible: {error}")


def draw_active_robot_marker(screen, robot):
    """Thick black frame around the robot you control."""
    rect = (robot.x * CELL_SIZE, robot.y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, (0, 0, 0), rect, 4)


def draw_hud(screen, font, world, active_robot):
    lines = [
        f"Score: {world.total_score()}/{world.total_objects()}    "
        f"Turn: {world.turn_count}/{world.max_turns}",
        f"Active robot: {active_robot.robot_id} ({active_robot.__class__.__name__})",
    ]
    if world.is_done():
        result = "SUCCESS" if world.is_success() else "time is up"
        lines.append(f"Game over: {result} (press R to play again)")

    for i, line in enumerate(lines):
        color = (200, 0, 0) if i == 2 else (0, 0, 0)
        screen.blit(font.render(line, True, color), (10, 10 + i * 25))


def handle_key(key, world, active_robot):
    """Apply one key press to the world. Every robot action costs one turn."""
    if world.is_done():
        return

    if key == pygame.K_SPACE:
        try_action(active_robot)
        world.increment_turn()

    elif key in ARROW_TO_DIRECTION:
        direction = ARROW_TO_DIRECTION[key]
        already_facing_this_way = active_robot.facing == direction
        active_robot.facing = direction
        # First press only turns; a second press in the same direction moves
        if already_facing_this_way and isinstance(active_robot, MobileRobot):
            active_robot.move(direction)
        world.increment_turn()


def main():
    pygame.init()
    font = pygame.font.SysFont(None, 24)

    world = create_world()
    all_robots = world.scouts + world.carriers + world.placers
    active_index = 0

    grid_width = world.grid.width * CELL_SIZE
    height = max(world.grid.height * CELL_SIZE + HUD_HEIGHT, 560)  # tall enough for the legend
    screen = pygame.display.set_mode((grid_width + LEGEND_WIDTH, height))
    pygame.display.set_caption("RoboRelay - keyboard game")

    running = True
    while running:
        active_robot = all_robots[active_index]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_TAB:
                    active_index = (active_index + 1) % len(all_robots)
                elif event.key == pygame.K_r:
                    world.reset()
                    # reset() creates NEW robot objects: rebuild our list
                    all_robots = world.scouts + world.carriers + world.placers
                    active_index = 0
                else:
                    handle_key(event.key, world, active_robot)

        active_robot = all_robots[active_index]
        screen.fill((255, 255, 255))

        # The grid is drawn below the HUD, in its own sub-area of the screen
        grid_area = screen.subsurface((0, HUD_HEIGHT, grid_width, world.grid.height * CELL_SIZE))
        draw_grid(grid_area, world)
        draw_robot_details(grid_area, world)
        draw_active_robot_marker(grid_area, active_robot)
        draw_hud(screen, font, world, active_robot)
        draw_legend(screen, grid_width + 15, 10, CONTROLS, show_active_marker=True)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()