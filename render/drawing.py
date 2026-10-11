"""
Drawing helpers shared by the keyboard game (play.py) and the agent viewer (watch_agent.py).

These functions only READ the world and draw it. No game rule lives here.
"""

import pygame

from env.cell_types import CellType
from robots.base_robot import DIRECTIONS

CELL_SIZE = 60  # default size of one cell, in pixels

COLORS = {
    CellType.EMPTY: (230, 230, 230),
    CellType.OBJECT: (255, 165, 0),
    CellType.SCOUT: (30, 144, 255),
    CellType.CARRIER: (34, 139, 34),
    CellType.PLACER: (220, 20, 60),
}


def draw_grid(screen, world, cell_size=CELL_SIZE):
    """Draw every cell as a colored square with a black border."""
    for y in range(world.grid.height):
        for x in range(world.grid.width):
            color = COLORS[world.grid.get(x, y)]
            rect = (x * cell_size, y * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (0, 0, 0), rect, 1)


def _cell_center(robot, cell_size):
    return (robot.x * cell_size + cell_size // 2, robot.y * cell_size + cell_size // 2)


def _draw_facing(screen, robot, cell_size):
    """White triangle pointing where the robot looks (pick_up, drop and charge act that way)."""
    cx, cy = _cell_center(robot, cell_size)
    dx, dy = DIRECTIONS[robot.facing]
    tip_dist, base_dist, half_width = cell_size * 0.40, cell_size * 0.22, cell_size * 0.12
    tip = (cx + dx * tip_dist, cy + dy * tip_dist)
    base1 = (cx + dx * base_dist - dy * half_width, cy + dy * base_dist - dx * half_width)
    base2 = (cx + dx * base_dist + dy * half_width, cy + dy * base_dist + dx * half_width)
    pygame.draw.polygon(screen, (255, 255, 255), [tip, base1, base2])


def draw_robot_details(screen, world, cell_size=CELL_SIZE):
    """Draw what colors alone do not show: where robots look, carried items, stored count."""
    for robot in world.scouts + world.carriers:
        _draw_facing(screen, robot, cell_size)

    # Carried item: a small orange square in the middle of the Carrier
    for carrier in world.carriers:
        if carrier.carrying_object:
            cx, cy = _cell_center(carrier, cell_size)
            half = int(cell_size * 0.14)
            box = (cx - half, cy - half, 2 * half, 2 * half)
            pygame.draw.rect(screen, (255, 165, 0), box)
            pygame.draw.rect(screen, (0, 0, 0), box, 2)

    # Number of stored items, written on each Placer
    font = pygame.font.SysFont(None, int(cell_size * 0.36))
    for placer in world.placers:
        text = font.render(str(placer.stored_count), True, (255, 255, 255))
        screen.blit(text, text.get_rect(center=_cell_center(placer, cell_size)))



# --- Legend (side panel) ---------------------------------------------------

LEGEND_WIDTH = 290  # width of the side panel, in pixels

_SWATCH = 26        # size of a small example square in the legend
_LINE = 32          # vertical space between two legend lines


def _draw_swatch(screen, left, top, kind):
    """Draw a small example of one visual element, like a mini cell."""
    rect = pygame.Rect(left, top, _SWATCH, _SWATCH)

    if kind in (CellType.SCOUT, CellType.CARRIER, CellType.PLACER, CellType.OBJECT):
        pygame.draw.rect(screen, COLORS[kind], rect)
        pygame.draw.rect(screen, (0, 0, 0), rect, 1)

    elif kind == "facing":
        pygame.draw.rect(screen, COLORS[CellType.CARRIER], rect)
        cx, cy = rect.center
        pygame.draw.polygon(screen, (255, 255, 255), [(cx + 10, cy), (cx + 2, cy - 6), (cx + 2, cy + 6)])

    elif kind == "carrying":
        pygame.draw.rect(screen, COLORS[CellType.CARRIER], rect)
        box = pygame.Rect(0, 0, 10, 10)
        box.center = rect.center
        pygame.draw.rect(screen, COLORS[CellType.OBJECT], box)
        pygame.draw.rect(screen, (0, 0, 0), box, 1)

    elif kind == "stored":
        pygame.draw.rect(screen, COLORS[CellType.PLACER], rect)
        font = pygame.font.SysFont(None, 20)
        text = font.render("2", True, (255, 255, 255))
        screen.blit(text, text.get_rect(center=rect.center))

    elif kind == "active":
        pygame.draw.rect(screen, COLORS[CellType.EMPTY], rect)
        pygame.draw.rect(screen, (0, 0, 0), rect, 3)


def draw_legend(screen, left, top, controls, show_active_marker=False):
    """Side panel: what each color and symbol means, then the keyboard controls.

    controls: list of (key, meaning) pairs, different for each window.
    """
    title_font = pygame.font.SysFont(None, 26, bold=True)
    font = pygame.font.SysFont(None, 22)
    y = top

    screen.blit(title_font.render("Legend", True, (0, 0, 0)), (left, y))
    y += _LINE

    items = [
        (CellType.SCOUT, "Scout: explores (not used yet)"),
        (CellType.CARRIER, "Carrier: moves, carries items"),
        (CellType.PLACER, "Placer: fixed, stores items"),
        (CellType.OBJECT, "Item to store"),
        ("facing", "Direction the robot looks"),
        ("carrying", "Carrier holding an item"),
        ("stored", "Number of stored items"),
    ]
    if show_active_marker:
        items.append(("active", "Robot you control"))

    for kind, label in items:
        _draw_swatch(screen, left, y, kind)
        screen.blit(font.render(label, True, (0, 0, 0)), (left + _SWATCH + 10, y + 6))
        y += _LINE

    y += 12
    screen.blit(title_font.render("Controls", True, (0, 0, 0)), (left, y))
    y += _LINE

    for key, meaning in controls:
        key_text = font.render(key, True, (0, 0, 0))
        key_box = pygame.Rect(left, y, max(_SWATCH, key_text.get_width() + 12), _SWATCH)
        pygame.draw.rect(screen, (235, 235, 235), key_box, border_radius=4)
        pygame.draw.rect(screen, (120, 120, 120), key_box, 1, border_radius=4)
        screen.blit(key_text, key_text.get_rect(center=key_box.center))
        screen.blit(font.render(meaning, True, (0, 0, 0)), (key_box.right + 10, y + 6))
        y += _LINE