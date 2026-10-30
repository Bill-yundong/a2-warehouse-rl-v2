# -*- coding: utf-8 -*-
"""
COSC-4117EL Assignment 2 - Group 1
PART 2 (and the Q-Learning side of PART 3):
    Model-free tabular Q-Learning for the warehouse robot task.

The Q-Learning agent uses the SAME environment and stochastic transition
behaviour as Part 1 (intended action with probability p, perpendicular slips
with (1-p)/2 each), but it has NO access to P(s'|s,a): it must discover the
policy purely from sampled experience.

Algorithm (per step, epsilon-greedy):
    a ~ epsilon-greedy(Q, s)
    take a, observe (s', r, done) from the environment
    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * (1-done) * max_a' Q(s',a') - Q(s,a) ]

Experiments:
  * Two hyperparameter configurations at p = 0.80 (alpha / gamma / epsilon
    decay), each trained over several random seeds (robustness).
  * PART 3: the chosen configuration is also trained for p = 1.00 / 0.90 /
    0.80 and compared with the Value Iteration policies stored in
    results/vi_results.json (if that file is missing, the VI script is run
    automatically).

How to run (from the folder containing this file):
    python COSC_4117EL_A2_G1-q_learning.py

Outputs:
    figures/ql_learning_curves.png
    figures/ql_policy_p080.png
    figures/ql_demo_trajectory_p080.png
    figures/part3_comparison.png
    results/ql_results.json

Generative AI note: implementation was drafted with assistance from Doubao
(ByteDance AI). See the report for the full AI-use statement and prompts.
"""

import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
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

# Reuse the VI plotting/policy utilities so figures use the same visual style.
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_VI_PATH = os.path.join(_HERE, "COSC_4117EL_A2_G1-value_iteration.py")
_spec = importlib.util.spec_from_file_location("value_iteration_a2", _VI_PATH)
vi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vi)  # type: ignore[union-attr]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LAYOUT_SEED = 99
MAX_STEPS = 120
TRAINING_SEEDS = [42, 7, 123]
EVAL_EPISODES = 500
EVAL_SEED_START = 2000          # same evaluation seeds as the VI script

# Two meaningful hyperparameter configurations (PART 2 experiment).
CONFIGS = {
    "A_conservative": {
        "alpha": 0.1,
        "gamma": 0.95,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay": 0.9995,     # geometric per-episode decay
        "episodes": 8000,
    },
    "B_fast_adapting": {
        "alpha": 0.5,
        "gamma": 0.99,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay": 0.999,
        "episodes": 8000,
    },
}
CHOSEN_CONFIG = "A_conservative"     # used for the PART 3 comparison
P_VALUES = [1.00, 0.90, 0.80]
PART3_EPISODES = 6000

FIG_DIR = os.path.join(_HERE, "figures")
RES_DIR = os.path.join(_HERE, "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Q-Learning
# ---------------------------------------------------------------------------
def epsilon_greedy(Q: np.ndarray, state_idx: int, epsilon: float,
                   rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(len(ACTIONS)))
    # argmax with deterministic tie-breaking towards action index 0
    return int(np.argmax(Q[state_idx]))


def train_q_learning(world: WarehouseGridWorld, cfg: Dict, seed: int,
                     episodes: Optional[int] = None, verbose: bool = False):
    """Tabular Q-Learning with epsilon-greedy exploration.

    Returns the Q table and per-episode training statistics.
    """
    episodes = episodes or cfg["episodes"]
    alpha, gamma = cfg["alpha"], cfg["gamma"]
    eps_start, eps_end, eps_decay = cfg["eps_start"], cfg["eps_end"], cfg["eps_decay"]

    n_states = world.get_state_space_size()
    n_actions = len(ACTIONS)
    Q = np.zeros((n_states, n_actions), dtype=float)

    # Dedicated RNG for the agent, independent of the environment's slip RNG.
    agent_rng = np.random.default_rng(seed)
    world.reseed(seed * 31 + 100000)   # layout stays fixed; fresh slip RNG

    returns = np.zeros(episodes)
    successes = np.zeros(episodes, dtype=int)
    lengths = np.zeros(episodes, dtype=int)
    eps_history = np.zeros(episodes)

    for ep in range(episodes):
        epsilon = max(eps_end, eps_start * (eps_decay ** ep))
        eps_history[ep] = epsilon
        state = world.reset()
        s_idx = world.state_to_index(state)
        total = 0.0

        for _ in range(MAX_STEPS):
            a_idx = epsilon_greedy(Q, s_idx, epsilon, agent_rng)
            result = world.step(ACTIONS[a_idx])
            sp_idx = world.state_to_index(result.state)
            done = result.done
            target = result.reward + (0.0 if done else gamma * np.max(Q[sp_idx]))
            Q[s_idx, a_idx] += alpha * (target - Q[s_idx, a_idx])
            total += result.reward
            s_idx = sp_idx
            if done:
                if result.info.get("event") == "return_to_dock":
                    successes[ep] = 1
                break

        returns[ep] = total
        lengths[ep] = world.steps
        if verbose and (ep + 1) % max(1, episodes // 10) == 0:
            window = returns[max(0, ep - 99):ep + 1]
            print(f"    episode {ep + 1}/{episodes}: eps={epsilon:.3f}, "
                  f"mean return (last 100)={window.mean():.1f}")

    return {
        "Q": Q,
        "returns": returns,
        "successes": successes,
        "lengths": lengths,
        "eps_history": eps_history,
    }


def extract_greedy_policy(world: WarehouseGridWorld, Q: np.ndarray) -> Dict[tuple, str]:
    """argmax_a Q(s,a) over the same state set used by Value Iteration."""
    policy = {}
    for s in vi.enumerate_states(world):
        if vi.is_terminal(world, s):
            continue
        policy[s] = ACTIONS[int(np.argmax(Q[world.state_to_index(s)]))]
    return policy


def evaluate_q_policy(world: WarehouseGridWorld, policy: Dict[tuple, str]):
    """Greedy evaluation (exploration disabled). Reuses VI roll-out harness."""
    return vi.evaluate_policy(world, policy, EVAL_EPISODES, EVAL_SEED_START)


def rolling_mean(x: np.ndarray, window: int = 100) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(len(x)):
        lo = max(0, i - window + 1)
        out[i] = np.mean(x[lo:i + 1])
    return out


# ---------------------------------------------------------------------------
# Part 2: two hyperparameter configurations, multiple seeds, at p = 0.80
# ---------------------------------------------------------------------------
def run_part2():
    results = {"configs": {}, "training_seeds": TRAINING_SEEDS, "p": 0.80}
    trained = {}   # config -> seed -> training record

    for cfg_name, cfg in CONFIGS.items():
        print(f"[Part 2] Training config {cfg_name}: {cfg}")
        per_seed = []
        for seed in TRAINING_SEEDS:
            world = WarehouseGridWorld(seed=LAYOUT_SEED, max_steps=MAX_STEPS,
                                       intended_prob=0.8)
            record = train_q_learning(world, cfg, seed, verbose=True)
            policy = extract_greedy_policy(world, record["Q"])
            eval_stats = evaluate_q_policy(world, policy)
            per_seed.append({
                "seed": seed,
                "record": record,
                "policy": policy,
                "eval": eval_stats,
                "world": world,
            })
            print(f"  seed={seed}: eval avg return={eval_stats['avg_return']:.2f}, "
                  f"success={eval_stats['success_rate']:.3f}, "
                  f"avg steps={eval_stats['avg_steps']:.1f}")
        trained[cfg_name] = per_seed

        results["configs"][cfg_name] = {
            "cfg": cfg,
            "eval": {
                "avg_return_mean": float(np.mean([r["eval"]["avg_return"] for r in per_seed])),
                "avg_return_std": float(np.std([r["eval"]["avg_return"] for r in per_seed])),
                "success_rate_mean": float(np.mean([r["eval"]["success_rate"] for r in per_seed])),
                "avg_steps_mean": float(np.mean([r["eval"]["avg_steps"] for r in per_seed])),
                "per_seed": [
                    {"seed": r["seed"],
                     "avg_return": r["eval"]["avg_return"],
                     "success_rate": r["eval"]["success_rate"],
                     "avg_steps": r["eval"]["avg_steps"]}
                    for r in per_seed
                ],
            }
        }

    # ------------------------------------------------------------------
    # Learning-curve figure: cumulative reward & rolling success rate
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    colors = {"A_conservative": "#1f77b4", "B_fast_adapting": "#d62728"}
    for cfg_name, per_seed in trained.items():
        rets = np.stack([rolling_mean(r["record"]["returns"]) for r in per_seed])
        succ = np.stack([rolling_mean(r["record"]["successes"]) for r in per_seed])
        x = np.arange(rets.shape[1])
        for ax, data, ylabel in ((axes[0], rets, "episode return (100-episode rolling avg)"),
                                 (axes[1], succ, "task success rate (100-episode rolling avg)")):
            mean, std = data.mean(axis=0), data.std(axis=0)
            ax.plot(x, mean, color=colors[cfg_name], label=cfg_name)
            ax.fill_between(x, mean - std, mean + std, color=colors[cfg_name], alpha=0.18)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            ax.legend(loc="lower right")
    axes[1].set_xlabel("training episode")
    axes[0].set_title("Q-Learning learning curves (p = 0.80, mean +/- std over seeds "
                      f"{TRAINING_SEEDS})")
    axes[1].set_ylim(-0.05, 1.05)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "ql_learning_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)

    # Pick the seed with the best eval return for the chosen config; plot its
    # policy and one greedy evaluation trajectory.
    chosen_runs = trained[CHOSEN_CONFIG]
    best = max(chosen_runs, key=lambda r: r["eval"]["avg_return"])
    world = best["world"]
    policy = best["policy"]
    results["chosen_seed"] = best["seed"]

    # Reuse the VI panel drawing to render a Q-Learning policy figure.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    vi.draw_policy_panel(axes[0], world, policy, 0, 0)
    vi.draw_policy_panel(axes[1], world, policy, 1, 0)
    vi.draw_policy_panel(axes[2], world, policy, 1, 1)
    fig.suptitle(f"Q-Learning greedy policy (config {CHOSEN_CONFIG}, p = 0.80, "
                 f"gamma = {CONFIGS[CHOSEN_CONFIG]['gamma']}, seed = {best['seed']})",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = os.path.join(FIG_DIR, "ql_policy_p080.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)

    # Demonstration evaluation run (exploration disabled).
    traj = best["eval"]["demo_trajectory"]
    plot_trajectory(world, traj, path=os.path.join(FIG_DIR, "ql_demo_trajectory_p080.png"))

    return results, trained


def plot_trajectory(world: WarehouseGridWorld, trajectory: List[tuple], path: str):
    """Draw one successful greedy evaluation trajectory on the grid."""
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(9.5, -0.5)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, color="#cccccc", linewidth=0.8)

    for r in range(10):
        for c in range(10):
            cell = int(world.grid[r, c])
            color = {SHELF: "#141414", CONGESTION: "#e13c3c", PICKUP: "#4682ff",
                     PACKING: "#fadc28", DOCK: "#3cc878"}.get(cell)
            if color:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                           facecolor=color, edgecolor="none"))
    for (tr, tc), lab in ((world.pickup_pos, "P"), (world.packing_pos, "D"),
                          (world.dock_pos, "C")):
        ax.text(tc, tr, lab, ha="center", va="center", fontsize=14,
                fontweight="bold", color="#111111")

    xs = [s[1] for s in trajectory]
    ys = [s[0] for s in trajectory]
    ax.plot(xs, ys, "-", color="#6a0dad", linewidth=2.2, alpha=0.8, zorder=3)
    ax.scatter(xs, ys, s=28, color="#6a0dad", zorder=4)
    # Mark phase transition points.
    for i, s in enumerate(trajectory):
        if i > 0 and (s[2], s[3]) != (trajectory[i - 1][2], trajectory[i - 1][3]):
            ax.scatter([s[1]], [s[0]], s=160, facecolors="none",
                       edgecolors="#ff8c00", linewidths=2.2, zorder=5)
    ax.scatter([xs[0]], [ys[0]], s=120, marker="o", facecolors="none",
               edgecolors="white", linewidths=2, zorder=5)
    ax.set_title(f"Q-Learning greedy evaluation run (p = 0.80, "
                 f"{len(trajectory) - 1} steps, successful)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)


# ---------------------------------------------------------------------------
# Part 3: stochastic comparison with Value Iteration
# ---------------------------------------------------------------------------
def policy_agreement(policy_a: Dict[tuple, str], policy_b: Dict[tuple, str],
                     keys: Optional[set] = None) -> float:
    if keys is None:
        keys = set(policy_a) & set(policy_b)
    else:
        keys = keys & set(policy_a) & set(policy_b)
    return float(np.mean([policy_a[k] == policy_b[k] for k in keys]))


def collect_visited_states(world: WarehouseGridWorld, policy: Dict[tuple, str],
                           n_episodes: int = EVAL_EPISODES,
                           seed_start: int = EVAL_SEED_START) -> set:
    """States actually reached by greedy roll-outs (the agent's reachable set)."""
    visited = set()
    for i in range(n_episodes):
        world.reseed(seed_start + i)
        state = world.reset()
        visited.add(state)
        while not world.is_terminal():
            state = world.step(policy.get(state, ACTIONS[0])).state
            visited.add(state)
    return visited


def run_part3(trained_part2) -> Dict:
    # Load VI results; run the VI script if the JSON is missing.
    vi_json = os.path.join(RES_DIR, "vi_results.json")
    if not os.path.exists(vi_json):
        print("VI results not found, running the Value Iteration script first...")
        subprocess.run([sys.executable, _VI_PATH], check=True, cwd=_HERE)
    with open(vi_json, "r", encoding="utf-8") as f:
        vi_summary = json.load(f)

    cfg = CONFIGS[CHOSEN_CONFIG]
    rows = []
    ql_policies = {}
    convergence_estimates = {}   # first episode after which rolling success >= 0.9

    for p in P_VALUES:
        print(f"[Part 3] Training Q-Learning under p = {p:.2f}")
        runs = []
        for seed in TRAINING_SEEDS:
            world = WarehouseGridWorld(seed=LAYOUT_SEED, max_steps=MAX_STEPS,
                                       intended_prob=p)
            record = train_q_learning(world, cfg, seed, episodes=PART3_EPISODES)
            policy = extract_greedy_policy(world, record["Q"])
            eval_stats = evaluate_q_policy(world, policy)
            runs.append({"seed": seed, "world": world, "policy": policy,
                         "eval": eval_stats, "record": record})
        ql_policies[p] = runs

        # VI policy for this p
        vi_pol_raw = vi_summary["policies"][f"{p:.2f}"]
        vi_policy = {tuple(map(int, k.split(","))): v for k, v in vi_pol_raw.items()}

        agreements = [policy_agreement(r["policy"], vi_policy) for r in runs]
        # Restrict the comparison to states the trained greedy policy actually
        # reaches in evaluation (rarely-visited cells keep arbitrary argmax).
        visited_agreements, visited_counts = [], []
        for r in runs:
            visited = collect_visited_states(r["world"], r["policy"])
            visited.discard(None)
            visited_counts.append(len(visited))
            visited_agreements.append(
                policy_agreement(r["policy"], vi_policy, keys=visited))
        mean_ret = float(np.mean([r["eval"]["avg_return"] for r in runs]))
        std_ret = float(np.std([r["eval"]["avg_return"] for r in runs]))
        mean_succ = float(np.mean([r["eval"]["success_rate"] for r in runs]))

        # Approximate convergence episode: first episode where rolling-100
        # success rate stays >= 0.9 for the remainder (using the median seed).
        median_run = sorted(runs, key=lambda r: r["eval"]["avg_return"])[len(runs) // 2]
        roll = rolling_mean(median_run["record"]["successes"])
        conv_ep = None
        for i in range(len(roll)):
            if np.all(roll[i:] >= 0.9):
                conv_ep = int(i)
                break
        convergence_estimates[p] = conv_ep

        vi_stats = vi_summary["runs"][f"{p:.2f}"]
        rows.append({
            "p": p,
            "vi_iterations": vi_stats["iterations_to_converge"],
            "vi_avg_return": vi_stats["avg_return"],
            "vi_success_rate": vi_stats["success_rate"],
            "vi_avg_steps": vi_stats["avg_steps"],
            "ql_avg_return_mean": mean_ret,
            "ql_avg_return_std": std_ret,
            "ql_success_rate_mean": mean_succ,
            "ql_avg_steps_mean": float(np.mean([r["eval"]["avg_steps"] for r in runs])),
            "ql_vi_policy_agreement_mean": float(np.mean(agreements)),
            "ql_vi_policy_agreement_std": float(np.std(agreements)),
            "ql_vi_policy_agreement_on_visited_mean": float(np.mean(visited_agreements)),
            "ql_vi_policy_agreement_on_visited_std": float(np.std(visited_agreements)),
            "n_visited_states": float(np.mean(visited_counts)),
            "ql_convergence_episode_median": conv_ep,
        })
        print(f"  p={p:.2f}: VI return={vi_stats['avg_return']:.2f} "
              f"success={vi_stats['success_rate']:.3f} | QL return={mean_ret:.2f} "
              f"success={mean_succ:.3f} | agreement={np.mean(agreements):.3f}")

    # ------------------------------------------------------------------
    # Comparison bar chart
    # ------------------------------------------------------------------
    ps = [r["p"] for r in rows]
    x = np.arange(len(ps))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    vi_ret = [r["vi_avg_return"] for r in rows]
    ql_ret = [r["ql_avg_return_mean"] for r in rows]
    ql_err = [r["ql_avg_return_std"] for r in rows]
    axes[0].bar(x - width / 2, vi_ret, width, label="Value Iteration", color="#2ca02c")
    axes[0].bar(x + width / 2, ql_ret, width, yerr=ql_err, capsize=4,
                label="Q-Learning", color="#1f77b4")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"p={p:.2f}" for p in ps])
    axes[0].set_ylabel("average cumulative return")
    axes[0].set_title("Average return (500 greedy eval episodes)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    vi_succ = [r["vi_success_rate"] for r in rows]
    ql_succ = [r["ql_success_rate_mean"] for r in rows]
    axes[1].bar(x - width / 2, vi_succ, width, label="Value Iteration", color="#2ca02c")
    axes[1].bar(x + width / 2, ql_succ, width, label="Q-Learning", color="#1f77b4")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"p={p:.2f}" for p in ps])
    axes[1].set_ylabel("task success rate")
    axes[1].set_ylim(0, 1.08)
    axes[1].set_title("Task success rate")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle("PART 3: Value Iteration vs Q-Learning under stochastic motion")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "part3_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)

    return {"comparison": rows}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    part2, trained = run_part2()
    part3 = run_part3(trained)

    summary = {
        "layout_seed": LAYOUT_SEED,
        "training_seeds": TRAINING_SEEDS,
        "eval_episodes": EVAL_EPISODES,
        "chosen_config": CHOSEN_CONFIG,
        "part2": part2,
        "part3": part3,
    }
    out_path = os.path.join(RES_DIR, "ql_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    print("saved", out_path)


if __name__ == "__main__":
    main()
