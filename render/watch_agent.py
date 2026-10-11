"""
Watch the trained Q-learning agent play, in a Pygame window.

    python3 -m render.watch_agent                              # watch forever
    python3 -m render.watch_agent --gif recordings/demo.gif    # record a GIF of 3 games

Controls:
    Space        pause / resume
    Right arrow  play one action (while paused)
    + / -        faster / slower
    Escape       quit
"""

import argparse
import os

import numpy as np
import pygame

from render.drawing import LEGEND_WIDTH, draw_grid, draw_legend, draw_robot_details
from rl.gym_env import RoboRelayEnv
from rl.q_learning import QLearningAgent

RECORDINGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "recordings")

CELL = 100           # bigger cells than the keyboard game: the grid is only 5x5
HUD_HEIGHT = 110     # info panel below the grid
END_PAUSE_MS = 1200  # how long the result stays on screen before the next game

CARRIER_ACTIONS = ["up", "down", "left", "right", "pick_up", "drop"]
PLACER_ACTIONS = ["wait", "charge"]

CONTROLS = [
    ("Space", "Pause / resume"),
    ("Right", "One action (paused)"),
    ("+ / -", "Faster / slower"),
    ("Esc", "Quit"),
]


def draw_hud(screen, font, env, game, last_actions, last_reward, game_reward, status):
    top = env.world.height * CELL
    pygame.draw.rect(screen, (250, 250, 250), (0, top, env.world.width * CELL, HUD_HEIGHT))
    world = env.world

    line1 = (f"Game {game}   |   Turn {world.turn_count}/{world.max_turns}   |   "
             f"Score {world.total_score()}/{world.total_objects()}")
    if last_actions is None:
        line2 = "Carrier: -    Placer: -"
    else:
        carrier_action, placer_action = last_actions
        line2 = (f"Carrier: {CARRIER_ACTIONS[carrier_action]}    "
                 f"Placer: {PLACER_ACTIONS[placer_action]}    reward {last_reward:+.2f}")
    line3 = f"Total reward: {game_reward:+.2f}    {status}"

    for i, line in enumerate([line1, line2, line3]):
        screen.blit(font.render(line, True, (0, 0, 0)), (10, top + 10 + i * 30))


def capture_frame(screen):
    """Turn the Pygame screen into a PIL image, for the GIF."""
    from PIL import Image
    pixels = pygame.surfarray.array3d(screen)                # shape (width, height, 3)
    return Image.fromarray(np.transpose(pixels, (1, 0, 2)))  # PIL wants (height, width, 3)


def main():
    parser = argparse.ArgumentParser(description="Watch the RoboRelay agent play.")
    parser.add_argument("--agent", default=os.path.join(RECORDINGS_DIR, "q_agent.pkl"))
    parser.add_argument("--speed", type=int, default=300, help="milliseconds between two actions")
    parser.add_argument("--gif", default=None, help="path of the GIF to record")
    parser.add_argument("--episodes", type=int, default=3, help="number of games to record in GIF mode")
    args = parser.parse_args()

    env = RoboRelayEnv()
    agent = QLearningAgent.load(args.agent)  # epsilon = 0: only what was learned, no random moves

    pygame.init()
    font = pygame.font.SysFont(None, 26)
    # The window is the grid plus the legend panel on the right
    grid_width = env.world.width * CELL
    screen = pygame.display.set_mode((grid_width + LEGEND_WIDTH, env.world.height * CELL + HUD_HEIGHT))
    pygame.display.set_caption("RoboRelay - Q-learning agent")
    clock = pygame.time.Clock()

    obs, _ = env.reset(randomize=True)
    game = 1
    last_actions, last_reward, game_reward = None, 0.0, 0.0
    done = False
    paused = False
    step_delay = args.speed
    step_timer = 0
    end_timer = 0

    frames = []
    frame_durations = []
    need_capture = True  # only capture a frame when the picture changes

    running = True
    while running:
        dt = clock.tick(60)  # milliseconds since the last loop
        step_now = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT and paused and not done:
                    step_now = True
                elif event.key in (pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_EQUALS):
                    step_delay = max(50, step_delay - 50)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    step_delay = min(2000, step_delay + 50)

        # --- Let the agent play at the chosen speed ---
        if not done and not paused:
            step_timer += dt
            if step_timer >= step_delay:
                step_timer = 0
                step_now = True

        if step_now:
            carrier_action, placer_action = agent.choose_action(obs)
            obs, reward, terminated, truncated, _ = env.step(np.array([carrier_action, placer_action]))
            last_actions = (carrier_action, placer_action)
            last_reward = reward
            game_reward += reward
            done = terminated or truncated
            need_capture = True

        # --- End of a game: show the result, then start a new one ---
        if done:
            end_timer += dt
            if end_timer >= END_PAUSE_MS:
                if args.gif and game >= args.episodes:
                    running = False
                else:
                    obs, _ = env.reset(randomize=True)
                    game += 1
                    last_actions, last_reward, game_reward = None, 0.0, 0.0
                    done = False
                    end_timer = 0
                    need_capture = True

        # --- Draw ---
        if done:
            status = "SUCCESS" if env.world.is_success() else "FAILED (time is up)"
        else:
            status = "PAUSED" if paused else ""

        screen.fill((255, 255, 255))
        draw_grid(screen, env.world, cell_size=CELL)
        draw_robot_details(screen, env.world, cell_size=CELL)
        draw_hud(screen, font, env, game, last_actions, last_reward, game_reward, status)
        draw_legend(screen, grid_width + 15, 10, CONTROLS)
        pygame.display.flip()

        if args.gif and need_capture:
            frames.append(capture_frame(screen))
            # the last frame of a game stays longer on screen
            frame_durations.append(END_PAUSE_MS if done else step_delay)
            need_capture = False

    pygame.quit()

    if args.gif and frames:
        frames[0].save(args.gif, save_all=True, append_images=frames[1:],
                       duration=frame_durations, loop=0)
        print(f"GIF saved: {args.gif} ({len(frames)} frames)")


if __name__ == "__main__":
    main()