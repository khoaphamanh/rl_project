import gymnasium as gym
import minigrid  # noqa: F401
import pygame
import sys

# ---------------------------------------------------------------------------
# MiniGrid encoding tables
# ---------------------------------------------------------------------------
IDX_TO_OBJ = {
    0: "unseen", 1: "empty", 2: "wall", 3: "floor",
    4: "door",   5: "key",   6: "ball", 7: "box",
    8: "goal",   9: "lava",  10: "agent",
}
IDX_TO_COLOR_NAME = {
    0: "red", 1: "green", 2: "blue",
    3: "purple", 4: "yellow", 5: "grey",
}
IDX_TO_STATE = {0: "open", 1: "closed", 2: "locked"}

# Objects whose display colour comes from the colour channel (not a fixed colour)
COLOR_DRIVEN = {4, 5, 6, 7}  # door, key, ball, box

# MiniGrid colour index → pygame RGB
MG_COLORS = [
    (220,  50,  50),  # 0 red
    ( 50, 200,  50),  # 1 green
    ( 60, 120, 220),  # 2 blue
    (160,  50, 220),  # 3 purple
    (240, 220,   0),  # 4 yellow
    (140, 140, 140),  # 5 grey
]

# Fixed display colours for objects that ignore the colour channel
OBJ_COLORS = {
    0:  ( 30,  30,  30),  # unseen
    1:  (210, 210, 210),  # empty
    2:  ( 75,  85, 105),  # wall
    3:  (185, 175, 145),  # floor
    8:  (  0, 200,  80),  # goal
    9:  (255,  90,   0),  # lava
    10: (255,  50,  50),  # agent
}

ACTION_NAMES = {
    0: "Turn Left",  1: "Turn Right", 2: "Move Forward",
    3: "Pick Up",    4: "Drop",       5: "Toggle",       6: "Done",
}

KEY_BINDINGS = [
    ("Arrow Left",  "Turn Left"),
    ("Arrow Right", "Turn Right"),
    ("Arrow Up",    "Move Forward"),
    ("P",           "Pick Up"),
    ("D",           "Drop"),
    ("Space",       "Toggle / open door"),
    ("Enter",       "Done"),
    ("R",           "Reset episode"),
    ("Q / Esc",     "Quit"),
]

SIDEBAR_WIDTH = 430
CELL_SIZE     = 35   # px per cell in partial-obs grid
FPS           = 30
ENV_ID        = "MiniGrid-MemoryS7-v0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_env():
    return gym.make(ENV_ID, render_mode="rgb_array")


def scale_frame(frame, h, w):
    surf = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
    return pygame.transform.scale(surf, (w, h))


def cell_bg(obj_idx, color_idx):
    if obj_idx in COLOR_DRIVEN:
        return MG_COLORS[color_idx] if color_idx < len(MG_COLORS) else (180, 180, 180)
    return OBJ_COLORS.get(obj_idx, (180, 180, 180))


def lum(r, g, b):
    return 0.299 * r + 0.587 * g + 0.114 * b


# Cell labels shown inside the grid squares
CELL_LABEL = {
    0: "",   1: "",  2: "W",   3: "",
    4: "D",  5: "K", 6: "●",  7: "[]",
    8: "G",  9: "!", 10: "▲",
}

# ---------------------------------------------------------------------------
# Partial-observation grid renderer
# ---------------------------------------------------------------------------

def draw_obs_grid(screen, image, ox, oy, font_tiny):
    """Draw the 7×7 encoded observation as coloured cells."""
    n = image.shape[0]          # 7
    agent_row = n - 1           # agent sits at bottom-centre
    agent_col = n // 2

    for row in range(n):
        for col in range(n):
            # MiniGrid obs is indexed [x, y] = [col, row], agent at [center, bottom]
            obj_idx, color_idx, _ = image[col, row]
            bg = cell_bg(obj_idx, color_idx)
            rect = pygame.Rect(ox + col * CELL_SIZE, oy + row * CELL_SIZE,
                               CELL_SIZE - 1, CELL_SIZE - 1)
            pygame.draw.rect(screen, bg, rect)

            # Yellow border on agent cell
            if row == agent_row and col == agent_col:
                pygame.draw.rect(screen, (255, 255, 80), rect, 2)

            label = CELL_LABEL.get(obj_idx, "")
            if label:
                r, g, b = bg
                txt_col = (20, 20, 20) if lum(r, g, b) > 128 else (240, 240, 240)
                surf = font_tiny.render(label, True, txt_col)
                screen.blit(surf, (rect.x + (CELL_SIZE - 1 - surf.get_width()) // 2,
                                   rect.y + (CELL_SIZE - 1 - surf.get_height()) // 2))

    # Forward arrow above agent cell
    ax = ox + agent_col * CELL_SIZE + CELL_SIZE // 2
    ay = oy + agent_row * CELL_SIZE - 2
    pygame.draw.polygon(screen, (255, 255, 80),
                        [(ax, ay - 7), (ax - 5, ay), (ax + 5, ay)])

    # Outer border
    pygame.draw.rect(screen, (110, 110, 110), (ox, oy, n * CELL_SIZE, n * CELL_SIZE), 1)


# ---------------------------------------------------------------------------
# Natural-language interpretation
# ---------------------------------------------------------------------------

def interpret_obs(image):
    """Return lines describing non-wall objects visible in the partial observation."""
    n = image.shape[0]
    agent_row = n - 1
    agent_col = n // 2
    lines = []

    for row in range(n):
        for col in range(n):
            # MiniGrid obs is indexed [x, y] = [col, row], agent at [center, bottom]
            obj_idx, color_idx, state = image[col, row]
            # skip walls, floor, empty, unseen, agent — walls are visible on the minimap
            if obj_idx in (0, 1, 2, 3, 10):
                continue

            steps_fwd = agent_row - row
            lateral   = col - agent_col

            side = ("center" if lateral == 0 else
                    f"{abs(lateral)} {'left' if lateral < 0 else 'right'}")
            if steps_fwd > 0:
                pos = f"{steps_fwd} step{'s' if steps_fwd > 1 else ''} ahead, {side}"
            elif steps_fwd == 0:
                pos = f"beside you, {side}"
            else:
                pos = f"{abs(steps_fwd)} step{'s' if abs(steps_fwd) > 1 else ''} behind, {side}"

            color_name = IDX_TO_COLOR_NAME.get(color_idx, "")
            obj_name   = IDX_TO_OBJ.get(obj_idx, "object")

            if obj_idx == 4:
                state_str = IDX_TO_STATE.get(state, "")
                lines.append(f"{color_name.capitalize()} door ({state_str}) — {pos}")
            elif obj_idx == 8:
                lines.append(f"GOAL tile — {pos}")
            elif obj_idx == 9:
                lines.append(f"LAVA — {pos}")
            else:
                lines.append(f"{color_name.capitalize()} {obj_name} — {pos}")

    # Action hint for whatever is directly in front of the agent
    if agent_row > 0:
        fi, fc, fs = image[agent_col, agent_row - 1]
        fname = IDX_TO_OBJ.get(fi, "?")
        fcol  = IDX_TO_COLOR_NAME.get(fc, "")
        if fi == 2:
            lines.insert(0, ">> Wall directly ahead — can't move forward")
        elif fi == 4:
            door_s = IDX_TO_STATE.get(fs, "")
            lines.insert(0, f">> {fcol.capitalize()} door ahead ({door_s})"
                            f" — press Space to {'open' if door_s != 'open' else 'close'}")
        elif fi in (5, 6, 7):
            lines.insert(0, f">> {fcol.capitalize()} {fname} directly ahead — press P to pick up")
        elif fi == 8:
            lines.insert(0, ">> GOAL tile directly ahead — move forward to reach it!")
        elif fi == 9:
            lines.insert(0, ">> LAVA directly ahead — stepping forward will end the episode!")

    if not lines:
        lines.append("Nothing notable visible in current view")

    return lines


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------

LEGEND = [
    ("▲",  (255, 255,  80), "you (agent)"),
    ("W",  ( 75,  85, 105), "wall"),
    ("D",  (140,  80,  20), "door"),
    ("K",  (240, 220,   0), "key"),
    ("●",  (220,  50,  50), "ball"),
    ("G",  (  0, 200,  80), "goal"),
    ("!",  (255,  90,   0), "lava"),
    ("[]", (200, 130,  50), "box"),
    ("",   ( 30,  30,  30), "unseen"),
]


def draw_legend(screen, ox, y, font_tiny):
    x = ox + 6
    for label, color, name in LEGEND:
        if x + 100 > ox + SIDEBAR_WIDTH - 6:
            x = ox + 6
            y += 18
        pygame.draw.rect(screen, color, (x, y, 13, 13))
        if label:
            r, g, b = color
            lc = (20, 20, 20) if lum(r, g, b) > 128 else (240, 240, 240)
            ls = font_tiny.render(label, True, lc)
            screen.blit(ls, (x + (13 - ls.get_width()) // 2,
                             y + (13 - ls.get_height()) // 2))
        t = font_tiny.render(f" {name}", True, (160, 160, 160))
        screen.blit(t, (x + 14, y))
        x += 14 + t.get_width() + 10
    return y + 18


# ---------------------------------------------------------------------------
# Full sidebar
# ---------------------------------------------------------------------------

def draw_sidebar(screen, font, small_font, font_tiny,
                 history, step, total_reward,
                 env_w, win_h, obs_image):
    sx = env_w
    pygame.draw.rect(screen, (26, 26, 26), (sx, 0, SIDEBAR_WIDTH, win_h))

    y   = 10
    pad = sx + 10

    def put(text, f=None, color=(210, 210, 210)):
        nonlocal y
        f = f or small_font
        s = f.render(text, True, color)
        screen.blit(s, (pad, y))
        y += s.get_height() + 3

    def divider():
        nonlocal y
        y += 5
        pygame.draw.line(screen, (65, 65, 65),
                         (sx + 5, y), (sx + SIDEBAR_WIDTH - 5, y))
        y += 7

    # ---- Info ----
    put(f"ENV: {ENV_ID}", font, (255, 210, 80))
    put(f"Step: {step}    Total reward: {total_reward:+.3f}", color=(160, 230, 160))
    divider()

    # ---- Partial observation grid ----
    put("PARTIAL OBSERVATION  (7×7 egocentric view)", font, (255, 210, 80))
    y += 3
    grid_px = 7 * CELL_SIZE
    grid_x  = sx + (SIDEBAR_WIDTH - grid_px) // 2
    draw_obs_grid(screen, obs_image, grid_x, y, font_tiny)
    y += grid_px + 8

    # legend
    y = draw_legend(screen, sx, y, font_tiny)
    y += 2
    divider()

    # ---- Interpretation ----
    put("WHAT YOU SEE", font, (255, 210, 80))
    y += 2
    for line in interpret_obs(obs_image):
        # soft word-wrap
        words   = line.split()
        row_str = ""
        for w in words:
            test = (row_str + " " + w).strip()
            if small_font.size(test)[0] > SIDEBAR_WIDTH - 20:
                put(row_str, color=(200, 200, 200))
                row_str = w
            else:
                row_str = test
        if row_str:
            put(row_str, color=(200, 200, 200))
    y += 4
    divider()

    # ---- Action history ----
    put("LAST ACTIONS", font, (255, 210, 80))
    y += 2
    for s, a, r, done in reversed(history[-6:]):
        col = (255, 100, 100) if done else (190, 190, 190)
        put(f"s{s:03d}  {a:<14} r={r:+.3f}{'  END' if done else ''}", color=col)

    # ---- Controls at bottom ----
    ctrl_y = win_h - len(KEY_BINDINGS) * 14 - 20
    pygame.draw.line(screen, (65, 65, 65),
                     (sx + 5, ctrl_y), (sx + SIDEBAR_WIDTH - 5, ctrl_y))
    ctrl_y += 6
    for key, desc in KEY_BINDINGS:
        t = font_tiny.render(f"{key:<14} {desc}", True, (120, 120, 120))
        screen.blit(t, (pad, ctrl_y))
        ctrl_y += t.get_height() + 2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    env = make_env()
    obs_state, _ = env.reset()
    current_obs  = obs_state["image"]   # shape (7, 7, 3): obj, color, state

    pygame.init()
    frame = env.render()
    env_h, env_w = frame.shape[:2]
    disp_h = max(env_h, 500)
    disp_w = max(env_w, 500)

    win_w = disp_w + SIDEBAR_WIDTH
    win_h = max(disp_h, 820)
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(f"MiniGrid Interactive — {ENV_ID}")
    clock = pygame.time.Clock()

    font       = pygame.font.SysFont("monospace", 13, bold=True)
    small_font = pygame.font.SysFont("monospace", 11)
    font_tiny  = pygame.font.SysFont("monospace", 10)

    step         = 0
    total_reward = 0.0
    history      = []
    terminated   = False
    truncated    = False
    message      = ""
    msg_timer    = 0

    KEY_MAP = {
        pygame.K_LEFT:   0,
        pygame.K_RIGHT:  1,
        pygame.K_UP:     2,
        pygame.K_p:      3,
        pygame.K_d:      4,
        pygame.K_SPACE:  5,
        pygame.K_RETURN: 6,
    }

    print("\nCONTROLS")
    print("--------")
    for k, d in KEY_BINDINGS:
        print(f"  {k:<14} {d}")
    print()

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
                    obs_state, _ = env.reset()
                    current_obs  = obs_state["image"]
                    step         = 0
                    total_reward = 0.0
                    history.clear()
                    terminated   = False
                    truncated    = False
                    message      = "Episode reset."
                    msg_timer    = FPS * 2

                elif event.key in KEY_MAP and not (terminated or truncated):
                    action = KEY_MAP[event.key]

        if action is not None:
            obs_state, reward, terminated, truncated, _ = env.step(action)
            current_obs   = obs_state["image"]
            step         += 1
            total_reward += reward
            done          = terminated or truncated
            history.append((step, ACTION_NAMES[action], reward, done))

            status = ("TERMINATED" if terminated else
                      "TRUNCATED"  if truncated  else "ongoing")
            print(f"Step {step:03d} | {ACTION_NAMES[action]:<14} | "
                  f"reward={reward:+.3f} | total={total_reward:+.3f} | {status}")

            if terminated or truncated:
                result  = "WON" if reward > 0 else "LOST"
                message = (f"EPISODE {result} — total reward={total_reward:.3f}  "
                           f"({'TERMINATED' if terminated else 'TRUNCATED'})  "
                           f"press R to restart")
                msg_timer = FPS * 9999

        # Draw env
        frame    = env.render()
        env_surf = scale_frame(frame, disp_h, disp_w)
        screen.fill((18, 18, 18))
        screen.blit(env_surf, (0, 0))

        draw_sidebar(screen, font, small_font, font_tiny,
                     history, step, total_reward,
                     disp_w, win_h, current_obs)

        if msg_timer > 0:
            ms = font.render(message, True, (255, 70, 70))
            screen.blit(ms, (10, disp_h - 28))
            msg_timer -= 1

        pygame.display.flip()
        clock.tick(FPS)

    env.close()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
