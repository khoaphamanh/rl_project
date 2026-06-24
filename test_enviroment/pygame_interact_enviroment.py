import gymnasium as gym
import minigrid  # noqa: F401
import pygame
import sys

# Action labels for display
ACTION_NAMES = {
    0: "Turn Left",
    1: "Turn Right",
    2: "Move Forward",
    3: "Pick Up",
    4: "Drop",
    5: "Toggle",
    6: "Done",
}

KEY_BINDINGS = """
CONTROLS
--------
Arrow Left  : Turn Left
Arrow Right : Turn Right
Arrow Up    : Move Forward
P           : Pick Up
D           : Drop
Space       : Toggle
Enter       : Done
R           : Reset
Q / Esc     : Quit
"""

SIDEBAR_WIDTH = 300
FPS = 30
ENV_ID = "MiniGrid-MemoryS7-v0"


def make_env():
    return gym.make(ENV_ID, render_mode="rgb_array")


def scale_frame(frame, target_h: int, target_w: int):
    """Scale a numpy RGB frame to target size using pygame."""
    surf = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
    return pygame.transform.scale(surf, (target_w, target_h))


def draw_sidebar(screen, font, small_font, history: list, step: int,
                 total_reward: float, env_w: int, win_h: int):
    sidebar_x = env_w
    pygame.draw.rect(screen, (30, 30, 30), (sidebar_x, 0, SIDEBAR_WIDTH, win_h))

    y = 10
    def put(text, f=None, color=(220, 220, 220), indent=0):
        nonlocal y
        f = f or small_font
        surf = f.render(text, True, color)
        screen.blit(surf, (sidebar_x + 10 + indent, y))
        y += surf.get_height() + 4

    put(f"ENV: {ENV_ID}", font, color=(255, 210, 80))
    y += 4
    put(f"Step : {step}", color=(180, 230, 180))
    put(f"Total reward : {total_reward:.3f}", color=(180, 230, 180))
    y += 8
    pygame.draw.line(screen, (80, 80, 80), (sidebar_x + 5, y), (sidebar_x + SIDEBAR_WIDTH - 5, y))
    y += 8

    put("LAST ACTIONS", font, color=(255, 210, 80))
    y += 2
    for entry in reversed(history[-14:]):
        s, a, r, done = entry
        color = (255, 100, 100) if done else (200, 200, 200)
        put(f"s{s:03d}  {a:<14} r={r:+.3f}{'  DONE' if done else ''}", color=color)

    y = win_h - 200
    pygame.draw.line(screen, (80, 80, 80), (sidebar_x + 5, y), (sidebar_x + SIDEBAR_WIDTH - 5, y))
    y += 8
    put("CONTROLS", font, color=(255, 210, 80))
    y += 2
    for line in KEY_BINDINGS.strip().splitlines()[2:]:
        put(line, color=(160, 160, 160))


def main():
    env = make_env()
    env.reset()

    pygame.init()
    frame = env.render()
    env_h, env_w = frame.shape[:2]
    env_display_h, env_display_w = max(env_h, 400), max(env_w, 400)

    win_w = env_display_w + SIDEBAR_WIDTH
    win_h = max(env_display_h, 600)
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(f"MiniGrid Interactive — {ENV_ID}")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("monospace", 14, bold=True)
    small_font = pygame.font.SysFont("monospace", 12)

    step = 0
    total_reward = 0.0
    history = []       # list of (step, action_name, reward, done)
    terminated = False
    truncated = False
    message = ""
    message_timer = 0

    KEY_TO_ACTION = {
        pygame.K_LEFT: 0,
        pygame.K_RIGHT: 1,
        pygame.K_UP: 2,
        pygame.K_p: 3,
        pygame.K_d: 4,
        pygame.K_SPACE: 5,
        pygame.K_RETURN: 6,
    }

    print(KEY_BINDINGS)

    running = True
    while running:
        action = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

                elif event.key == pygame.K_r:
                    env.reset()
                    step = 0
                    total_reward = 0.0
                    history.clear()
                    terminated = False
                    truncated = False
                    message = "Episode reset."
                    message_timer = FPS * 2

                elif event.key in KEY_TO_ACTION and not (terminated or truncated):
                    action = KEY_TO_ACTION[event.key]

        if action is not None:
            _, reward, terminated, truncated, _ = env.step(action)
            step += 1
            total_reward += reward
            done = terminated or truncated
            history.append((step, ACTION_NAMES[action], reward, done))

            print(
                f"Step {step:03d} | Action: {ACTION_NAMES[action]:<14} | "
                f"Reward: {reward:+.3f} | Total: {total_reward:+.3f} | "
                f"{'TERMINATED' if terminated else 'TRUNCATED' if truncated else 'ongoing'}"
            )

            if terminated:
                message = f"TERMINATED  total reward = {total_reward:.3f}  (press R to reset)"
                message_timer = FPS * 999
            elif truncated:
                message = f"TRUNCATED (max steps)  total reward = {total_reward:.3f}  (press R)"
                message_timer = FPS * 999

        # --- draw ---
        frame = env.render()
        env_surf = scale_frame(frame, env_display_h, env_display_w)
        screen.fill((20, 20, 20))
        screen.blit(env_surf, (0, 0))
        draw_sidebar(screen, font, small_font, history, step, total_reward, env_display_w, win_h)

        # overlay message
        if message_timer > 0:
            msg_surf = font.render(message, True, (255, 80, 80))
            screen.blit(msg_surf, (10, env_display_h - 30))
            message_timer -= 1

        pygame.display.flip()
        clock.tick(FPS)

    env.close()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
