# -*- coding: utf-8 -*-
"""
COSC-4117EL Assignment 2 - Group 1
OPTIONAL interactive viewer (not required by the assignment, not used for grading).

A pygame window that visualises the Value Iteration and Q-Learning policies on
the provided warehouse gridworld:

  * policy arrows and a value heat map for each task phase
  * switch the intended-action probability p = 1.00 / 0.90 / 0.80
  * animated greedy roll-out in the stochastic environment (slips are marked)
  * Q-Learning trained live, in small batches, with a learning curve
  * hover a cell to see Q(s, a) for the four actions

It only IMPORTS the existing code: the environment class from
``warehouse_gridworld_domain_random.py`` and the Value Iteration / Q-Learning
functions from the two algorithm scripts. Nothing in those files is changed and
no file in figures/ or results/ is written.

The Q-Learning loop below is the same update as ``train_q_learning`` in
COSC_4117EL_A2_G1-q_learning.py (config A, seed 42, same RNG seeding), only
split into batches so the window stays responsive. After 6000 episodes it
therefore reaches the same Q-table as the Part 3 run for that seed.

Usage:
    python COSC_4117EL_A2_G1-ui.py

Keys:
    V / Q        show Value Iteration / Q-Learning policy
    1 / 2 / 3    p = 1.00 / 0.90 / 0.80
    Tab          cycle the phase shown (Auto / Phase 1 / 2 / 3)
    A / H        toggle arrows / value heat map
    Space        play / pause the roll-out      N   single step
    R            new roll-out                    + / -  speed
    T            start / pause Q-Learning training for the current p
    E            evaluate the shown policy (200 greedy episodes)
    Esc          quit

Generative AI: this viewer was written with the assistance of Claude
(Anthropic). Conversation link: [paste Claude share link here]
"""

import importlib.util
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

from warehouse_gridworld_domain_random import (  # noqa: E402
    ACTIONS,
    CONGESTION,
    CONGESTION_COLOR,
    DOCK,
    DOCK_COLOR,
    EMPTY_COLOR,
    PACKING,
    PACKING_COLOR,
    PICKUP,
    PICKUP_COLOR,
    ROBOT_COLOR,
    SHELF,
    SHELF_COLOR,
    WarehouseGridWorld,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VI = _load("value_iteration_a2", "COSC_4117EL_A2_G1-value_iteration.py")
QL = _load("q_learning_a2", "COSC_4117EL_A2_G1-q_learning.py")

LAYOUT_SEED = VI.LAYOUT_SEED
MAX_STEPS = VI.MAX_STEPS
P_VALUES = [1.00, 0.90, 0.80]
QL_CFG = QL.CONFIGS[QL.CHOSEN_CONFIG]
QL_SEED = QL.TRAINING_SEEDS[0]          # 42
QL_TARGET_EPISODES = QL.PART3_EPISODES  # 6000
PHASES = [(0, 0), (1, 0), (1, 1)]
PHASE_NAMES = {(0, 0): "Phase 1: go to pickup P",
               (1, 0): "Phase 2: carry to packing D",
               (1, 1): "Phase 3: return to dock C"}

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
CELL = 60
GRID_X, GRID_Y = 20, 64
GRID_PX = 10 * CELL
PANEL_X = GRID_X + GRID_PX + 24
WIN_W, WIN_H = PANEL_X + 520, GRID_Y + GRID_PX + 60

BG = (246, 247, 250)
INK = (30, 34, 48)
MUTED = (110, 116, 130)
ACCENT = (36, 74, 126)
BTN = (226, 232, 241)
BTN_ON = (36, 74, 126)
GRID_LINE = (205, 208, 214)
ARROW_COLOR = (34, 34, 85)
SLIP_COLOR = (230, 120, 0)

ACTION_VEC = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class VIResult:
    def __init__(self, p: float):
        self.world = WarehouseGridWorld(seed=LAYOUT_SEED, max_steps=MAX_STEPS, intended_prob=p)
        t0 = time.perf_counter()
        self.V, self.policy, self.deltas, self.iterations = VI.value_iteration(self.world)
        self.seconds = time.perf_counter() - t0
        model = VI.build_model(self.world, VI.enumerate_states(self.world))
        self.Q: Dict[tuple, Dict[str, float]] = {}
        for s in self.policy:
            self.Q[s] = {a: sum(wr + pr * VI.GAMMA * self.V[ns] for pr, ns, wr, _ in model[s][a])
                         for a in ACTIONS}


class QLTrainer:
    """Incremental copy of QL.train_q_learning (same update and seeding)."""

    def __init__(self, p: float):
        self.world = WarehouseGridWorld(seed=LAYOUT_SEED, max_steps=MAX_STEPS, intended_prob=p)
        self.Q = np.zeros((self.world.get_state_space_size(), len(ACTIONS)))
        self.rng = np.random.default_rng(QL_SEED)
        self.world.reseed(QL_SEED * 31 + 100000)
        self.returns: List[float] = []
        self.successes: List[int] = []
        self.episode = 0
        self.training = False

    @property
    def epsilon(self) -> float:
        return max(QL_CFG["eps_end"], QL_CFG["eps_start"] * (QL_CFG["eps_decay"] ** self.episode))

    def run_episode(self) -> None:
        alpha, gamma = QL_CFG["alpha"], QL_CFG["gamma"]
        eps = self.epsilon
        state = self.world.reset()
        s = self.world.state_to_index(state)
        total, success = 0.0, 0
        for _ in range(MAX_STEPS):
            a = QL.epsilon_greedy(self.Q, s, eps, self.rng)
            result = self.world.step(ACTIONS[a])
            sp = self.world.state_to_index(result.state)
            target = result.reward + (0.0 if result.done else gamma * np.max(self.Q[sp]))
            self.Q[s, a] += alpha * (target - self.Q[s, a])
            total += result.reward
            s = sp
            if result.done:
                success = int(result.info.get("event") == "return_to_dock")
                break
        self.returns.append(total)
        self.successes.append(success)
        self.episode += 1

    def train_for(self, budget_s: float) -> None:
        t_end = time.perf_counter() + budget_s
        while time.perf_counter() < t_end:
            self.run_episode()
            if self.episode >= QL_TARGET_EPISODES and self.episode % QL_TARGET_EPISODES == 0:
                self.training = False
                break

    def policy(self) -> Dict[tuple, str]:
        return QL.extract_greedy_policy(self.world, self.Q)

    def q_of(self, s: tuple) -> Dict[str, float]:
        row = self.Q[self.world.state_to_index(s)]
        return {a: float(v) for a, v in zip(ACTIONS, row)}


# ---------------------------------------------------------------------------
# Small UI helpers
# ---------------------------------------------------------------------------
class Button:
    def __init__(self, rect, label, action, key_hint=""):
        self.rect = pygame.Rect(rect)
        self.label, self.action, self.key_hint = label, action, key_hint
        self.on = False

    def draw(self, surf, font):
        color = BTN_ON if self.on else BTN
        pygame.draw.rect(surf, color, self.rect, border_radius=6)
        txt = font.render(self.label, True, (255, 255, 255) if self.on else INK)
        surf.blit(txt, txt.get_rect(center=self.rect.center))


def lerp(c1, c2, t):
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def heat_color(t: float):
    """t in [0, 1]: low = warm red, mid = pale, high = green."""
    lo, mid, hi = (226, 110, 96), (250, 246, 232), (84, 176, 120)
    return lerp(lo, mid, t * 2) if t < 0.5 else lerp(mid, hi, (t - 0.5) * 2)


def rolling(x: List[float], w: int = 100) -> np.ndarray:
    if not x:
        return np.array([])
    a = np.asarray(x, dtype=float)
    c = np.cumsum(np.insert(a, 0, 0.0))
    idx = np.arange(1, len(a) + 1)
    lo = np.maximum(0, idx - w)
    return (c[idx] - c[lo]) / (idx - lo)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
class App:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Warehouse robot - Value Iteration vs Q-Learning")
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.clock = pygame.time.Clock()
        self.f_title = pygame.font.SysFont("arial", 22, bold=True)
        self.f = pygame.font.SysFont("arial", 15)
        self.f_b = pygame.font.SysFont("arial", 15, bold=True)
        self.f_s = pygame.font.SysFont("arial", 13)
        self.f_cell = pygame.font.SysFont("arial", 11)

        self.method = "VI"
        self.p = 0.80
        self.phase_mode = "auto"  # or index 0..2
        self.show_arrows = True
        self.show_heat = True
        self.playing = False
        self.speed = 6.0  # steps per second
        self.eval_text: Dict[Tuple[str, float], str] = {}
        self.message = ""

        self.vi: Dict[float, VIResult] = {}
        self.ql: Dict[float, QLTrainer] = {p: QLTrainer(p) for p in P_VALUES}
        self._ql_policy_cache: Dict[float, Tuple[int, Dict[tuple, str]]] = {}
        self._loading_screen("Running Value Iteration for p = 1.00 / 0.90 / 0.80 ...")
        for p in P_VALUES:
            self.vi[p] = VIResult(p)

        self._build_buttons()
        self.rollout_seed = 2000
        self.new_rollout()

    # ------------------------------------------------------------ setup
    def _loading_screen(self, text):
        self.screen.fill(BG)
        t = self.f_title.render(text, True, INK)
        self.screen.blit(t, t.get_rect(center=(WIN_W // 2, WIN_H // 2)))
        pygame.display.flip()

    def _build_buttons(self):
        x, y = PANEL_X, GRID_Y
        self.btn_method = [Button((x + 90, y, 70, 28), "VI", lambda: self.set_method("VI")),
                           Button((x + 166, y, 110, 28), "Q-Learning", lambda: self.set_method("QL"))]
        y += 36
        self.btn_p = [Button((x + 90 + i * 76, y, 70, 28), f"{p:.2f}", (lambda p=p: self.set_p(p)))
                      for i, p in enumerate(P_VALUES)]
        y += 36
        labels = ["Auto", "1 Pickup", "2 Deliver", "3 Return"]
        widths = [56, 80, 84, 80]
        self.btn_phase, bx = [], x + 90
        for i, (lab, w) in enumerate(zip(labels, widths)):
            mode = "auto" if i == 0 else i - 1
            self.btn_phase.append(Button((bx, y, w, 28), lab, (lambda m=mode: self.set_phase(m))))
            bx += w + 6
        y += 36
        self.btn_arrows = Button((x + 90, y, 80, 28), "Arrows", self.toggle_arrows)
        self.btn_heat = Button((x + 176, y, 100, 28), "Heat map", self.toggle_heat)
        y += 44
        self.btn_play = Button((x, y, 80, 30), "Play", self.toggle_play)
        self.btn_step = Button((x + 86, y, 64, 30), "Step", self.step_once)
        self.btn_reset = Button((x + 156, y, 92, 30), "New run", self.new_rollout)
        self.btn_slow = Button((x + 254, y, 30, 30), "-", lambda: self.change_speed(0.5))
        self.btn_fast = Button((x + 290, y, 30, 30), "+", lambda: self.change_speed(2.0))
        self.btn_eval = Button((x + 330, y, 150, 30), "Evaluate policy", self.evaluate)
        self.btn_train = Button((x, 0, 150, 30), "Train Q-Learning", self.toggle_train)
        self.buttons = (self.btn_method + self.btn_p + self.btn_phase +
                        [self.btn_arrows, self.btn_heat, self.btn_play, self.btn_step,
                         self.btn_reset, self.btn_slow, self.btn_fast, self.btn_eval, self.btn_train])

    # ------------------------------------------------------------ actions
    def set_method(self, m):
        self.method = m
        self.new_rollout()

    def set_p(self, p):
        self.p = p
        self.new_rollout()

    def set_phase(self, m):
        self.phase_mode = m

    def toggle_arrows(self):
        self.show_arrows = not self.show_arrows

    def toggle_heat(self):
        self.show_heat = not self.show_heat

    def toggle_play(self):
        if self.env.done:
            self.new_rollout()
        self.playing = not self.playing

    def change_speed(self, f):
        self.speed = float(np.clip(self.speed * f, 0.75, 96.0))

    def toggle_train(self):
        tr = self.ql[self.p]
        tr.training = not tr.training
        if tr.training:
            self.method = "QL"

    def current_policy(self) -> Dict[tuple, str]:
        if self.method == "VI":
            return self.vi[self.p].policy
        tr = self.ql[self.p]
        cached = self._ql_policy_cache.get(self.p)
        if cached is None or cached[0] != tr.episode:
            cached = (tr.episode, tr.policy())
            self._ql_policy_cache[self.p] = cached
        return cached[1]

    def values_for_phase(self, phase) -> Dict[Tuple[int, int], float]:
        out = {}
        if self.method == "VI":
            V = self.vi[self.p].V
            for (r, c, h, d), v in V.items():
                if (h, d) == phase:
                    out[(r, c)] = v
        else:
            tr = self.ql[self.p]
            for s in VI.enumerate_states(tr.world):
                if (s[2], s[3]) == phase and not VI.is_terminal(tr.world, s):
                    out[(s[0], s[1])] = float(np.max(tr.Q[tr.world.state_to_index(s)]))
        return out

    def q_values(self, s) -> Optional[Dict[str, float]]:
        if self.method == "VI":
            return self.vi[self.p].Q.get(s)
        return self.ql[self.p].q_of(s)

    def new_rollout(self):
        self.env = WarehouseGridWorld(seed=LAYOUT_SEED, max_steps=MAX_STEPS, intended_prob=self.p)
        self.rollout_seed += 1
        self.env.reseed(self.rollout_seed)
        self.state = self.env.reset()
        self.trail = [(self.state[0], self.state[1])]
        self.slips: List[Tuple[int, int]] = []
        self.log: List[str] = []
        self.last_event = "start"
        self.playing = False
        self._acc = 0.0

    def step_once(self):
        if self.env.done:
            return
        policy = self.current_policy()
        action = policy.get(self.state, ACTIONS[0])
        res = self.env.step(action)
        realized = res.info.get("realized_action", action)
        self.state = res.state
        self.trail.append((self.state[0], self.state[1]))
        ev = res.info.get("event", "move")
        if realized != action:
            self.slips.append((self.state[0], self.state[1]))
        line = f"t={self.env.steps:3d}  {action:>5s}"
        line += f" -> slipped {realized}" if realized != action else ""
        line += f"   r={res.reward:+.0f}"
        if ev not in ("move",):
            line += f"   [{ev}]"
        self.log.append(line)
        self.last_event = ev
        if res.done:
            self.playing = False

    def evaluate(self, n: int = 200):
        policy = self.current_policy()
        env = WarehouseGridWorld(seed=LAYOUT_SEED, max_steps=MAX_STEPS, intended_prob=self.p)
        rets, succ, steps = [], [], []
        for i in range(n):
            env.reseed(VI.EVAL_SEED_START + i)
            s = env.reset()
            total = 0.0
            ok = 0
            while not env.is_terminal():
                r = env.step(policy.get(s, ACTIONS[0]))
                s = r.state
                total += r.reward
                if r.info.get("event") == "return_to_dock":
                    ok = 1
            rets.append(total)
            succ.append(ok)
            steps.append(env.steps)
        self.eval_text[(self.method, self.p)] = (
            f"{n} greedy runs: avg return {np.mean(rets):.2f}, success {np.mean(succ):.0%}, "
            f"avg steps {np.mean(steps):.1f}")

    # ------------------------------------------------------------ drawing
    def shown_phase(self):
        if self.phase_mode == "auto":
            return (self.state[2], self.state[3])
        return PHASES[self.phase_mode]

    def cell_rect(self, r, c):
        return pygame.Rect(GRID_X + c * CELL, GRID_Y + r * CELL, CELL, CELL)

    def draw_grid(self):
        world = self.env
        phase = self.shown_phase()
        values = self.values_for_phase(phase) if self.show_heat else {}
        if values:
            vmin, vmax = min(values.values()), max(values.values())
            span = (vmax - vmin) or 1.0
        for r in range(10):
            for c in range(10):
                rect = self.cell_rect(r, c)
                cell = int(world.grid[r, c])
                color = {SHELF: SHELF_COLOR, CONGESTION: CONGESTION_COLOR, PICKUP: PICKUP_COLOR,
                         PACKING: PACKING_COLOR, DOCK: DOCK_COLOR}.get(cell, EMPTY_COLOR)
                if cell == 0 and (r, c) in values:
                    color = heat_color((values[(r, c)] - vmin) / span)
                pygame.draw.rect(self.screen, color, rect)
                if cell != SHELF and cell != 0 and (r, c) in values:
                    # stations / congestion keep their colour; show value as a strip
                    t = (values[(r, c)] - vmin) / span
                    pygame.draw.rect(self.screen, heat_color(t), (rect.x + 3, rect.bottom - 9, CELL - 6, 6))
                label = {CONGESTION: "!", PICKUP: "P", PACKING: "D", DOCK: "C"}.get(cell)
                if label:
                    t = self.f_b.render(label, True, (255, 255, 255) if cell == CONGESTION else INK)
                    self.screen.blit(t, (rect.x + 4, rect.y + 2))
        for i in range(11):
            pygame.draw.line(self.screen, GRID_LINE, (GRID_X + i * CELL, GRID_Y),
                             (GRID_X + i * CELL, GRID_Y + GRID_PX))
            pygame.draw.line(self.screen, GRID_LINE, (GRID_X, GRID_Y + i * CELL),
                             (GRID_X + GRID_PX, GRID_Y + i * CELL))

        if self.show_arrows:
            policy = self.current_policy()
            for (r, c, h, d), a in policy.items():
                if (h, d) != phase:
                    continue
                self.draw_arrow(r, c, a)

        # trail and slips
        if len(self.trail) > 1:
            pts = [self.cell_rect(r, c).center for r, c in self.trail]
            pygame.draw.lines(self.screen, (120, 40, 200), False, pts, 3)
        for (r, c) in self.slips:
            cx, cy = self.cell_rect(r, c).center
            pygame.draw.line(self.screen, SLIP_COLOR, (cx - 7, cy - 7), (cx + 7, cy + 7), 3)
            pygame.draw.line(self.screen, SLIP_COLOR, (cx - 7, cy + 7), (cx + 7, cy - 7), 3)
        # robot
        rr, rc = self.state[0], self.state[1]
        center = self.cell_rect(rr, rc).center
        pygame.draw.circle(self.screen, ROBOT_COLOR, center, CELL // 3)
        if self.state[2]:
            pygame.draw.rect(self.screen, (250, 220, 40), (center[0] - 6, center[1] - 6, 12, 12))

        # hover info
        mx, my = pygame.mouse.get_pos()
        if GRID_X <= mx < GRID_X + GRID_PX and GRID_Y <= my < GRID_Y + GRID_PX:
            c, r = (mx - GRID_X) // CELL, (my - GRID_Y) // CELL
            pygame.draw.rect(self.screen, ACCENT, self.cell_rect(r, c), 3)
            self.hover = (r, c) + phase
        else:
            self.hover = None

    def draw_arrow(self, r, c, a):
        cx, cy = self.cell_rect(r, c).center
        dx, dy = ACTION_VEC[a]
        L, W = CELL * 0.30, CELL * 0.13
        tip = (cx + dx * L, cy + dy * L)
        tail = (cx - dx * L * 0.8, cy - dy * L * 0.8)
        base = (cx + dx * L * 0.15, cy + dy * L * 0.15)
        pygame.draw.line(self.screen, ARROW_COLOR, tail, base, 3)
        px, py = -dy, dx
        pygame.draw.polygon(self.screen, ARROW_COLOR,
                            [tip, (base[0] + px * W, base[1] + py * W), (base[0] - px * W, base[1] - py * W)])

    def text(self, s, pos, font=None, color=INK):
        self.screen.blit((font or self.f).render(s, True, color), pos)

    def draw_panel(self):
        x = PANEL_X
        # sync button states
        for b, m in zip(self.btn_method, ["VI", "QL"]):
            b.on = self.method == m
        for b, p in zip(self.btn_p, P_VALUES):
            b.on = abs(self.p - p) < 1e-9
        for i, b in enumerate(self.btn_phase):
            b.on = (self.phase_mode == "auto") if i == 0 else (self.phase_mode == i - 1)
        self.btn_arrows.on, self.btn_heat.on = self.show_arrows, self.show_heat
        self.btn_play.label = "Pause" if self.playing else "Play"
        self.btn_play.on = self.playing
        tr = self.ql[self.p]
        self.btn_train.on = tr.training
        self.btn_train.label = "Pause training" if tr.training else "Train Q-Learning"

        self.text("Method", (x, GRID_Y + 6), self.f_b)
        self.text("p", (x, GRID_Y + 42), self.f_b)
        self.text("Phase", (x, GRID_Y + 78), self.f_b)
        self.text("Overlay", (x, GRID_Y + 114), self.f_b)

        y = GRID_Y + 196
        # model info
        if self.method == "VI":
            v = self.vi[self.p]
            start = (v.world.dock_pos[0], v.world.dock_pos[1], 0, 0)
            info = [f"Value Iteration  (gamma = {VI.GAMMA}, theta = {VI.THETA:g})",
                    f"converged in {v.iterations} sweeps ({v.seconds:.2f} s),  V*(start) = {v.V[start]:.2f}"]
        else:
            info = [f"Q-Learning config {QL.CHOSEN_CONFIG}: alpha {QL_CFG['alpha']}, gamma {QL_CFG['gamma']},"
                    f" eps decay {QL_CFG['eps_decay']}, seed {QL_SEED}",
                    f"episodes trained: {tr.episode}   epsilon: {tr.epsilon:.3f}"
                    + ("   (untrained - press Train)" if tr.episode == 0 else "")]
        for i, line in enumerate(info):
            self.text(line, (x, y + i * 19), self.f_s, ACCENT if i == 0 else INK)
        y += 44
        ev = self.eval_text.get((self.method, self.p))
        if ev:
            self.text(ev, (x, y), self.f_s, MUTED)
        y += 22

        # roll-out status
        status = (f"Roll-out (p = {self.p:.2f}, seed {self.rollout_seed}):  step {self.env.steps}"
                  f"   score {self.env.score:.0f}   package {'yes' if self.state[2] else 'no'}"
                  f"   delivered {'yes' if self.state[3] else 'no'}")
        self.text(status, (x, y), self.f_s)
        y += 18
        if self.env.done:
            done = "SUCCESS - back at the dock" if self.last_event == "return_to_dock" else "stopped: step limit"
            self.text(done, (x, y), self.f_b, (40, 140, 70) if "SUCCESS" in done else (190, 60, 50))
        else:
            self.text(f"showing: {PHASE_NAMES[self.shown_phase()]}   speed {self.speed:g} steps/s", (x, y),
                      self.f_s, MUTED)
        y += 20
        for line in self.log[-5:]:
            self.text(line, (x, y), self.f_cell, SLIP_COLOR if "slipped" in line else MUTED)
            y += 14
        y = max(y, GRID_Y + 362)

        # hover Q values
        if self.hover is not None:
            r, c, h, d = self.hover
            q = self.q_values(self.hover) if not self.env.grid[r, c] == SHELF else None
            if q:
                best = max(q.values())
                parts = "   ".join(f"{a} {v:7.2f}{'*' if abs(v - best) < 1e-6 else ''}" for a, v in q.items())
                self.text(f"cell ({r},{c}) phase {PHASES.index((h, d)) + 1}:  {parts}", (x, y), self.f_cell)
        y += 20

        # training button + curve
        self.btn_train.rect.topleft = (x, y)
        self.text(f"target {QL_TARGET_EPISODES} episodes per p (same as Part 3)", (x + 160, y + 8), self.f_s, MUTED)
        y += 38
        self.draw_curve(pygame.Rect(x, y, 490, WIN_H - y - 40), tr)

        for b in self.buttons:
            b.draw(self.screen, self.f)

    def draw_curve(self, rect, tr: QLTrainer):
        pygame.draw.rect(self.screen, (255, 255, 255), rect)
        pygame.draw.rect(self.screen, GRID_LINE, rect, 1)
        self.text(f"Q-Learning learning curve, p = {self.p:.2f} (100-episode rolling mean)",
                  (rect.x + 6, rect.y + 4), self.f_s, ACCENT)
        plot = pygame.Rect(rect.x + 44, rect.y + 24, rect.w - 86, rect.h - 44)
        ymin, ymax = -400.0, 150.0
        n_max = max(QL_TARGET_EPISODES, tr.episode)

        def px(i):
            return plot.x + plot.w * i / max(1, n_max - 1)

        def py(v):
            return plot.bottom - plot.h * (np.clip(v, ymin, ymax) - ymin) / (ymax - ymin)

        for v in (-400, -200, 0, 100):
            yy = py(v)
            pygame.draw.line(self.screen, (236, 238, 242), (plot.x, yy), (plot.right, yy))
            self.text(f"{v}", (rect.x + 4, yy - 7), self.f_cell, MUTED)
        for frac in (0.0, 0.5, 1.0):
            yy = plot.bottom - plot.h * frac
            self.text(f"{frac:.0%}", (plot.right + 6, yy - 7), self.f_cell, (60, 150, 90))
        self.text("return", (plot.x, plot.bottom + 4), self.f_cell, ACCENT)
        self.text("success rate", (plot.x + 60, plot.bottom + 4), self.f_cell, (60, 150, 90))
        self.text(f"episode {tr.episode}/{n_max}", (plot.right - 110, plot.bottom + 4), self.f_cell, MUTED)
        # reference: VI evaluation return if available
        if tr.episode > 1:
            ret = rolling(tr.returns)
            suc = rolling(tr.successes)
            step = max(1, len(ret) // plot.w)
            idx = range(0, len(ret), step)
            pts = [(px(i), py(ret[i])) for i in idx]
            spts = [(px(i), plot.bottom - plot.h * suc[i]) for i in idx]
            if len(pts) > 1:
                pygame.draw.lines(self.screen, (60, 150, 90), False, spts, 2)
                pygame.draw.lines(self.screen, ACCENT, False, pts, 2)

    def draw(self):
        self.screen.fill(BG)
        self.text("Warehouse robot: Value Iteration vs Q-Learning", (GRID_X, 18), self.f_title)
        self.text("V/Q method  1/2/3 p  Tab phase  A arrows  H heat map  Space play  N step  R new run"
                  "  T train  E evaluate  +/- speed", (GRID_X, GRID_Y + GRID_PX + 14), self.f_s, MUTED)
        self.text("Hover a cell to see Q(s,a). Orange x = the move slipped to a perpendicular direction.",
                  (GRID_X, GRID_Y + GRID_PX + 32), self.f_s, MUTED)
        self.draw_grid()
        self.draw_panel()

    # ------------------------------------------------------------ loop
    def handle_key(self, key):
        mapping = {
            pygame.K_v: lambda: self.set_method("VI"),
            pygame.K_q: lambda: self.set_method("QL"),
            pygame.K_1: lambda: self.set_p(1.00),
            pygame.K_2: lambda: self.set_p(0.90),
            pygame.K_3: lambda: self.set_p(0.80),
            pygame.K_a: self.toggle_arrows,
            pygame.K_h: self.toggle_heat,
            pygame.K_SPACE: self.toggle_play,
            pygame.K_n: self.step_once,
            pygame.K_r: self.new_rollout,
            pygame.K_t: self.toggle_train,
            pygame.K_e: self.evaluate,
            pygame.K_EQUALS: lambda: self.change_speed(2.0),
            pygame.K_PLUS: lambda: self.change_speed(2.0),
            pygame.K_KP_PLUS: lambda: self.change_speed(2.0),
            pygame.K_MINUS: lambda: self.change_speed(0.5),
            pygame.K_KP_MINUS: lambda: self.change_speed(0.5),
            pygame.K_TAB: self.cycle_phase,
        }
        if key in mapping:
            mapping[key]()

    def cycle_phase(self):
        order = ["auto", 0, 1, 2]
        self.phase_mode = order[(order.index(self.phase_mode) + 1) % len(order)]

    def update(self, dt):
        tr = self.ql[self.p]
        if tr.training:
            tr.train_for(0.03)
        if self.playing:
            self._acc += dt * self.speed
            while self._acc >= 1.0 and self.playing:
                self._acc -= 1.0
                self.step_once()

    def run(self):
        self.hover = None
        while True:
            dt = self.clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    self.handle_key(event.key)
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for b in self.buttons:
                        if b.rect.collidepoint(event.pos):
                            b.action()
                            break
            self.update(dt)
            self.draw()
            pygame.display.flip()


def snapshot(out_dir: str) -> None:
    """Headless self-test: render a few states to PNG files (used for testing)."""
    os.makedirs(out_dir, exist_ok=True)
    app = App()
    app.hover = None
    app.set_p(0.80)
    for _ in range(14):
        app.step_once()
    app.evaluate()
    app.draw()
    pygame.image.save(app.screen, os.path.join(out_dir, "ui_vi_p080.png"))
    app.set_method("QL")
    tr = app.ql[0.80]
    while tr.episode < QL_TARGET_EPISODES:
        tr.run_episode()
    for _ in range(40):
        app.step_once()
    app.evaluate()
    app.draw()
    pygame.image.save(app.screen, os.path.join(out_dir, "ui_ql_p080.png"))
    np.save(os.path.join(out_dir, "q_table_p080_seed42.npy"), tr.Q)
    pygame.quit()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--snapshot":
        snapshot(sys.argv[2])
    else:
        App().run()
