# -*- coding: utf-8 -*-
"""
COSC-4117EL Assignment 2 - Group 1
PART 1 (and the Value Iteration side of PART 3):
    Model-based Dynamic Programming for the warehouse robot MDP.

The transition model P(s'|s,a) and reward model R(s,a,s') are KNOWN. The
environment module ``warehouse_gridworld_domain_random`` exposes a pure
transition kernel; this script enumerates the full MDP over
    states  s = (row, col, has_package, delivered)
    actions a in {up, down, left, right}
and applies Value Iteration:

    V_{k+1}(s) = max_a sum_{s'} P(s'|s,a) [ R(s,a,s') + gamma * V_k(s') ]

until max_s |V_{k+1}(s) - V_k(s)| < theta, then extracts the greedy policy.

It runs for intended-action probabilities p = 1.00, 0.90, 0.80 (PART 3),
evaluates each policy by Monte Carlo roll-outs in the *same* stochastic
environment, saves policy-arrow figures, a convergence figure, and a JSON
results file that the Q-Learning script reuses for the PART 3 comparison.

How to run (from the folder containing this file):
    python COSC_4117EL_A2_G1-value_iteration.py

Outputs:
    figures/vi_policy_p100.png, vi_policy_p090.png, vi_policy_p080.png
    figures/vi_convergence.png
    results/vi_results.json

Generative AI note: implementation was drafted with assistance from Doubao
(ByteDance AI). See the report for the full AI-use statement and prompts.
"""

import json
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # headless rendering, no display required
import matplotlib.pyplot as plt
import numpy as np

from warehouse_gridworld_domain_random import (
    ACTIONS,
    CONGESTION,
    DOCK,
    PACKING,
    PICKUP,
    SHELF,
    WarehouseGridWorld,
)

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
LAYOUT_SEED = 99          # fixed layout (matches Figure 1 of the assignment)
GAMMA = 0.95              # discount factor reported in the report
THETA = 1e-6              # convergence threshold on max |V_{k+1} - V_k|
P_VALUES = [1.00, 0.90, 0.80]          # PART 3 stochastic settings
EVAL_EPISODES = 500                     # Monte Carlo roll-outs per policy
EVAL_SEED_START = 2000                  # deterministic, reproducible eval seeds
MAX_STEPS = 120

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
RES_DIR = os.path.join(HERE, "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

ARROW = {"up": "\u2191", "down": "\u2193", "left": "\u2190", "right": "\u2192"}
CELL_COLORS = {
    SHELF: "#141414",
    CONGESTION: "#e13c3c",
    PICKUP: "#4682ff",
    PACKING: "#fadc28",
    DOCK: "#3cc878",
}
PHASE_TITLES = {
    (0, 0): "Phase 1: no package (go to pickup P)",
    (1, 0): "Phase 2: carrying package (go to packing D)",
    (1, 1): "Phase 3: delivered (return to dock C)",
}


# ---------------------------------------------------------------------------
# MDP construction
# ---------------------------------------------------------------------------
def enumerate_states(world: WarehouseGridWorld) -> List[Tuple[int, int, int, int]]:
    """All reachable task states: passable cells x the three task phases.

    (has_package=0, delivered=1) is logically impossible (delivery requires
    the package), so that flag combination is excluded from the MDP.
    """
    states = []
    for r in range(world.size):
        for c in range(world.size):
            if world.grid[r, c] == SHELF:
                continue
            for has_package, delivered in ((0, 0), (1, 0), (1, 1)):
                states.append((r, c, has_package, delivered))
    return states


def is_terminal(world: WarehouseGridWorld, state: Tuple[int, int, int, int]) -> bool:
    """Success terminal: back at the charging dock after delivery.

    The 120-step timeout is an episode horizon for simulation only, not a
    stationary MDP state, so it is not part of the VI model.
    """
    r, c, _, delivered = state
    return (r, c) == world.dock_pos and delivered == 1


def build_model(world: WarehouseGridWorld, states):
    """Enumerate P(s'|s,a) and the expected reward of each aggregated outcome.

    For each intended action, sum the environment kernel outcomes over the
    realized-direction distribution {p: intended, (1-p)/2: each perpendicular}.
    Multiple realized directions may produce the same s' (e.g. a blocked slip
    leaves the robot in place), so outcomes are aggregated by next state.

    Returns:
        q_sa[state][action] = list of
            (P(s'|s,a), next_state, P(s'|s,a) * E[R|s,a,s'], done)
    so the Bellman backup is simply sum over outcomes of
        (weighted_reward + prob * gamma * V[next_state]).
    """
    """Like build_model, but each outcome stores prob * expected reward.

    outcome tuple: (prob, next_state, prob_weighted_expected_reward, done)
    """
    q_sa = {}
    for s in states:
        q_sa[s] = {}
        for intended in ACTIONS:
            merged: Dict[tuple, Dict[str, float]] = {}
            for realized, prob in world.action_probabilities(intended).items():
                ns, reward, done, _ = world.transition_kernel(s, realized, 0)
                entry = merged.setdefault(ns, {"prob": 0.0, "rew": 0.0, "done": done})
                entry["prob"] += prob
                entry["rew"] += prob * reward
                entry["done"] = entry["done"] or done
            q_sa[s][intended] = [
                (v["prob"], ns, v["rew"], v["done"]) for ns, v in merged.items()
            ]
    return q_sa


# ---------------------------------------------------------------------------
# Value Iteration
# ---------------------------------------------------------------------------
def value_iteration(world: WarehouseGridWorld, gamma: float = GAMMA,
                    theta: float = THETA):
    """In-place-free Value Iteration with the Bellman optimality backup."""
    states = enumerate_states(world)
    model = build_model(world, states)
    terminal = {s for s in states if is_terminal(world, s)}

    V = {s: 0.0 for s in states}
    policy: Dict[tuple, str] = {}
    deltas: List[float] = []
    iteration = 0

    while True:
        new_V = dict(V)
        delta = 0.0
        for s in states:
            if s in terminal:
                new_V[s] = 0.0
                continue
            best = -np.inf
            for a in ACTIONS:
                q = 0.0
                for pr, ns, weighted_reward, _ in model[s][a]:
                    q += weighted_reward + pr * gamma * V[ns]
                if q > best:
                    best = q
            new_V[s] = best
            delta = max(delta, abs(best - V[s]))
        V = new_V
        iteration += 1
        deltas.append(delta)
        if delta < theta:
            break

    # Greedy policy extraction (ties: fixed ACTIONS order for reproducibility).
    for s in states:
        if s in terminal:
            continue
        best_a, best_q = ACTIONS[0], -np.inf
        for a in ACTIONS:
            q = 0.0
            for pr, ns, weighted_reward, _ in model[s][a]:
                q += weighted_reward + pr * gamma * V[ns]
            if q > best_q:
                best_q, best_a = q, a
        policy[s] = best_a

    return V, policy, deltas, iteration


# ---------------------------------------------------------------------------
# Policy evaluation inside the live stochastic environment
# ---------------------------------------------------------------------------
def evaluate_policy(world: WarehouseGridWorld, policy: Dict[tuple, str],
                    n_episodes: int = EVAL_EPISODES,
                    seed_start: int = EVAL_SEED_START):
    """Run greedy roll-outs with exploration disabled.

    Returns average return, success rate, average length, and the trajectory
    of one successful episode for visualization.
    """
    returns, successes, lengths = [], [], []
    demo_trajectory = None

    for i in range(n_episodes):
        world.reseed(seed_start + i)          # layout unchanged, new RNG
        state = world.reset()
        total = 0.0
        trajectory = [state]
        success = False
        while not world.is_terminal():
            action = policy.get(state, ACTIONS[0])
            result = world.step(action)
            state = result.state
            total += result.reward
            trajectory.append(state)
            if result.info.get("event") == "return_to_dock":
                success = True
        returns.append(total)
        successes.append(int(success))
        lengths.append(world.steps)
        if demo_trajectory is None and success:
            demo_trajectory = trajectory

    return {
        "avg_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "success_rate": float(np.mean(successes)),
        "avg_steps": float(np.mean(lengths)),
        "demo_trajectory": demo_trajectory,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def draw_policy_panel(ax, world: WarehouseGridWorld, policy, has_package: int,
                      delivered: int, highlight=None):
    """Draw one task-phase grid with arrows for the greedy policy."""
    ax.set_xlim(-0.5, world.size - 0.5)
    ax.set_ylim(world.size - 0.5, -0.5)
    ax.set_xticks(range(world.size))
    ax.set_yticks(range(world.size))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#888888")
    ax.grid(True, color="#cccccc", linewidth=0.8)

    for r in range(world.size):
        for c in range(world.size):
            cell = int(world.grid[r, c])
            if cell == SHELF:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                           facecolor=CELL_COLORS[SHELF],
                                           edgecolor="none"))
            elif cell == CONGESTION:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                           facecolor=CELL_COLORS[CONGESTION],
                                           edgecolor="none"))
                ax.text(c - 0.30, r - 0.30, "!", ha="center", va="center",
                        color="white", fontsize=9, fontweight="bold")
            else:
                color = CELL_COLORS.get(cell, None)
                if color is not None:
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                               facecolor=color, edgecolor="none"))

    # Station labels (small, in the corner so the policy arrow stays visible)
    for (tr, tc), lab in ((world.pickup_pos, "P"), (world.packing_pos, "D"),
                          (world.dock_pos, "C")):
        ax.text(tc - 0.30, tr - 0.30, lab, ha="center", va="center",
                fontsize=10, fontweight="bold", color="#111111")

    # Policy arrows (centred in each cell)
    for (r, c, h, d), action in policy.items():
        if h != has_package or d != delivered:
            continue
        if is_terminal(world, (r, c, h, d)):
            continue
        ax.text(c, r + 0.08, ARROW[action], ha="center", va="center",
                fontsize=15, color="#222255", fontweight="bold")

    ax.set_title(PHASE_TITLES[(has_package, delivered)], fontsize=11)

    if highlight is not None:
        hr, hc = highlight
        ax.add_patch(plt.Rectangle((hc - 0.5, hr - 0.5), 1, 1, fill=False,
                                   edgecolor="#ff00ff", linewidth=3.0))


def save_policy_figure(world, policy, p: float, highlight=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    draw_policy_panel(axes[0], world, policy, 0, 0, highlight)
    draw_policy_panel(axes[1], world, policy, 1, 0, highlight)
    draw_policy_panel(axes[2], world, policy, 1, 1, highlight)
    fig.suptitle(f"Value Iteration policy (intended-action probability p = {p:.2f}, "
                 f"gamma = {GAMMA})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    tag = f"{p:.2f}".replace(".", "")
    path = os.path.join(FIG_DIR, f"vi_policy_p{tag}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main experiment: PART 1 + PART 3 (Value Iteration side)
# ---------------------------------------------------------------------------
def run():
    summary = {
        "layout_seed": LAYOUT_SEED,
        "gamma": GAMMA,
        "theta": THETA,
        "p_values": P_VALUES,
        "pickup": None,
        "packing": None,
        "dock": None,
        "runs": {},
    }

    policies = {}
    all_deltas = {}

    for p in P_VALUES:
        world = WarehouseGridWorld(seed=LAYOUT_SEED, max_steps=MAX_STEPS,
                                   intended_prob=p)
        if summary["pickup"] is None:
            summary["pickup"] = list(world.pickup_pos)
            summary["packing"] = list(world.packing_pos)
            summary["dock"] = list(world.dock_pos)

        V, policy, deltas, n_iter = value_iteration(world)
        policies[p] = policy
        all_deltas[p] = deltas
        eval_stats = evaluate_policy(world, policy)

        tag = f"{p:.2f}"
        summary["runs"][tag] = {
            "iterations_to_converge": n_iter,
            "final_max_delta": deltas[-1],
            "value_at_start": V[world.dock_pos + (0, 0)],
            "avg_return": eval_stats["avg_return"],
            "std_return": eval_stats["std_return"],
            "success_rate": eval_stats["success_rate"],
            "avg_steps": eval_stats["avg_steps"],
        }
        fig_path = save_policy_figure(world, policy, p)
        print(f"[p={tag}] VI iterations={n_iter}, final delta={deltas[-1]:.2e}, "
              f"V(start)={V[world.dock_pos + (0, 0)]:.2f}, "
              f"avg return={eval_stats['avg_return']:.2f}, "
              f"success={eval_stats['success_rate']:.3f}, "
              f"avg steps={eval_stats['avg_steps']:.1f}; saved {fig_path}")

        # Save a demo trajectory for the p=0.8 setting (used in PART 1 report).
        if abs(p - 0.8) < 1e-9:
            summary["demo_trajectory_p080"] = [list(s) for s in eval_stats["demo_trajectory"]]

    # ------------------------------------------------------------------
    # PART 3: states whose preferred action changes as p decreases
    # ------------------------------------------------------------------
    changes = []
    pol_100, pol_080 = policies[1.0], policies[0.8]
    for s in pol_100:
        if pol_100[s] != pol_080[s]:
            r, c, h, d = s
            # Is the cell (or an immediate neighbour) a congestion zone?
            near_congestion = any(
                0 <= r + dr < 10 and 0 <= c + dc < 10
                and int(world.grid[r + dr, c + dc]) == CONGESTION
                for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
            )
            changes.append({
                "state": list(s),
                "action_p1.00": pol_100[s],
                "action_p0.90": policies[0.9][s],
                "action_p0.80": pol_080[s],
                "near_congestion": near_congestion,
            })
    summary["policy_changes_1.00_to_0.80"] = changes
    summary["n_policy_changes"] = len(changes)
    congestion_changes = [c for c in changes if c["near_congestion"]]
    summary["n_policy_changes_near_congestion"] = len(congestion_changes)

    # Highlight one change near a congestion zone on the p=0.8 figure.
    highlight = None
    if congestion_changes:
        r, c, _, _ = congestion_changes[0]["state"]
        highlight = (r, c)
        summary["highlighted_change"] = congestion_changes[0]
        world08 = WarehouseGridWorld(seed=LAYOUT_SEED, max_steps=MAX_STEPS,
                                     intended_prob=0.8)
        save_policy_figure(world08, policies[0.8], 0.8, highlight=highlight)
    print(f"Policy states changing between p=1.00 and p=0.80: {len(changes)} "
          f"({len(congestion_changes)} at/adjacent to congestion)")
    for ch in changes[:10]:
        print("  ", ch)

    # ------------------------------------------------------------------
    # Convergence figure: max Bellman residual per sweep (log scale)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for p in P_VALUES:
        ax.plot(range(1, len(all_deltas[p]) + 1), all_deltas[p],
                label=f"p = {p:.2f}")
    ax.axhline(THETA, color="grey", linestyle="--", linewidth=1,
               label=f"theta = {THETA:g}")
    ax.set_yscale("log")
    ax.set_xlabel("Value Iteration sweep")
    ax.set_ylabel(r"max$_s$ |V$_{k+1}$(s) - V$_k$(s)| (log scale)")
    ax.set_title("Value Iteration convergence under different stochastic settings")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    conv_path = os.path.join(FIG_DIR, "vi_convergence.png")
    fig.tight_layout()
    fig.savefig(conv_path, dpi=150)
    plt.close(fig)
    print("saved", conv_path)

    # ------------------------------------------------------------------
    # Policy serialised for the Q-Learning comparison script
    # ------------------------------------------------------------------
    summary["policies"] = {
        f"{p:.2f}": {",".join(map(str, s)): a for s, a in pol.items()}
        for p, pol in policies.items()
    }
    out_path = os.path.join(RES_DIR, "vi_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("saved", out_path)
    return summary


if __name__ == "__main__":
    run()
