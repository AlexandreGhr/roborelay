import pygame

from env.cell_types import CellType
from env.world import World
from robots.scout import Scout
from robots.carrier import Carrier
from robots.placer import Placer

CELL_SIZE = 60

COLORS = {
    CellType.EMPTY: (230, 230, 230),
    CellType.OBJECT: (255, 165, 0),
    CellType.SCOUT: (30, 144, 255),
    CellType.CARRIER: (34, 139, 34),
    CellType.PLACER: (220, 20, 60),
}

ARROW_TO_DIRECTION = {
    pygame.K_UP: "up",
    pygame.K_DOWN: "down",
    pygame.K_LEFT: "left",
    pygame.K_RIGHT: "right",
}


def draw_grid(screen, world):
    for y in range(world.grid.height):
        for x in range(world.grid.width):
            cell_type = world.grid.get(x, y)
            color = COLORS[cell_type]

            rect = (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (0, 0, 0), rect, 1)


def draw_active_robot_marker(screen, robot):
    center_x = robot.x * CELL_SIZE + CELL_SIZE // 2
    center_y = robot.y * CELL_SIZE + CELL_SIZE // 2
    pygame.draw.circle(screen, (0, 0, 0), (center_x, center_y), 8)


def draw_hud(screen, font, world, active_robot):
    score_text = font.render(f"Score : {world.total_score()}", True, (0, 0, 0))
    screen.blit(score_text, (10, 10))

    robot_text = font.render(
        f"Robot actif : {active_robot.robot_id} ({active_robot.__class__.__name__}) "
        f"- regarde : {active_robot.facing}",
        True,
        (0, 0, 0),
    )
    screen.blit(robot_text, (10, 35))

    turn_text = font.render(
        f"Tour : {world.turn_count} / {world.max_turns}", True, (0, 0, 0)
    )
    screen.blit(turn_text, (300, 10))

    if world.is_done():
        if world.total_score() >= world.total_objects():
            message = "Partie terminée : VICTOIRE ! (appuie sur R pour rejouer)"
        else:
            message = "Partie terminée : temps écoulé (appuie sur R pour rejouer)"
        end_text = font.render(message, True, (200, 0, 0))
        screen.blit(end_text, (10, 35 + 25))


def try_action(robot):
    try:
        if isinstance(robot, Carrier):
            if robot.carrying_object:
                robot.drop(robot.facing)
            else:
                robot.pick_up(robot.facing)
        elif isinstance(robot, Placer):
            robot.charge(robot.facing)
    except ValueError as e:
        print(f"Action impossible : {e}")


if __name__ == "__main__":
    pygame.init()
    font = pygame.font.SysFont(None, 24)

    world = World(10, 8)
    world.add_scout("scout-1", 0, 0)
    world.add_carrier("carrier-1", 2, 2)
    world.add_placer("placer-1", 8, 6)
    world.add_object(4, 3)
    world.add_object(6, 5)

    all_robots = world.scouts + world.carriers + world.placers
    active_index = 0

    window_width = world.grid.width * CELL_SIZE
    window_height = world.grid.height * CELL_SIZE + 60
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("RoboRelay")

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
                    all_robots = world.scouts + world.carriers + world.placers
                    active_index = 0

                elif not world.is_done():
                    if event.key == pygame.K_SPACE:
                        try_action(active_robot)
                        world.increment_turn()

                    elif event.key in ARROW_TO_DIRECTION:
                        direction = ARROW_TO_DIRECTION[event.key]
                        already_facing_this_way = (active_robot.facing == direction)
                        active_robot.facing = direction

                        if already_facing_this_way and isinstance(active_robot, (Scout, Carrier)):
                            active_robot.move(direction)
                        world.increment_turn()
        screen.fill((255, 255, 255))

        grid_surface = screen.subsurface((0, 60, window_width, window_height - 60))
        draw_grid(grid_surface, world)
        draw_active_robot_marker(grid_surface, active_robot)

        draw_hud(screen, font, world, active_robot)

        pygame.display.flip()

    pygame.quit()