# -*- coding: utf-8 -*-
"""Supplementary route analysis (referenced by the report):

For each intended-action probability p and each method (Value Iteration /
Q-Learning), run greedy roll-outs and count how often the robot enters a
congestion zone, how many (realised) moves are blocked, and the average path
length. This supports the PART 3 discussion of whether uncertainty changes
the preferred route near congestion zones.

Usage:
    python route_analysis.py
Writes results/route_analysis.json.
"""
import json
import os

import numpy as np

from warehouse_gridworld_domain_random import (
    ACTIONS, CONGESTION, WarehouseGridWorld,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(HERE, "results")
LAYOUT_SEED = 99
P_VALUES = [1.00, 0.90, 0.80]
EVAL_EPISODES = 500
EVAL_SEED_START = 2000


def load_policies():
    with open(os.path.join(RES_DIR, "vi_results.json"), encoding="utf-8") as f:
        vi_summary = json.load(f)
    vi_policies = {
        float(p): {tuple(map(int, k.split(","))): a for k, a in pol.items()}
        for p, pol in vi_summary["policies"].items()
    }
    # Q-Learning policies are not serialised per seed; re-train quickly here
    # through the Q-Learning module (config A, median seed 7) for the same p.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ql", os.path.join(HERE, "COSC_4117EL_A2_G1-q_learning.py"))
    ql = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ql)
    ql_policies = {}
    for p in P_VALUES:
        world = WarehouseGridWorld(seed=LAYOUT_SEED, max_steps=120, intended_prob=p)
        rec = ql.train_q_learning(
            world, ql.CONFIGS[ql.CHOSEN_CONFIG], seed=7,
            episodes=ql.PART3_EPISODES)
        ql_policies[p] = (world, ql.extract_greedy_policy(world, rec["Q"]))
    return vi_policies, ql_policies


def rollout_stats(world, policy):
    congestion_entries, blocked, lengths, successes = [], [], [], []
    for i in range(EVAL_EPISODES):
        world.reseed(EVAL_SEED_START + i)
        state = world.reset()
        cong = blocked_moves = 0
        success = False
        while not world.is_terminal():
            action = policy.get(state, ACTIONS[0])
            result = world.step(action)
            if result.info.get("congestion"):
                cong += 1
            if result.info.get("event") == "invalid_move":
                blocked_moves += 1
            state = result.state
            if result.info.get("event") == "return_to_dock":
                success = True
        congestion_entries.append(cong)
        blocked.append(blocked_moves)
        lengths.append(world.steps)
        successes.append(int(success))
    return {
        "mean_congestion_entries": float(np.mean(congestion_entries)),
        "mean_blocked_moves": float(np.mean(blocked)),
        "mean_length": float(np.mean(lengths)),
        "success_rate": float(np.mean(successes)),
    }


def main():
    vi_policies, ql_policies = load_policies()
    out = {}
    for p in P_VALUES:
        world_vi = WarehouseGridWorld(seed=LAYOUT_SEED, max_steps=120,
                                      intended_prob=p)
        vi_stats = rollout_stats(world_vi, vi_policies[p])
        world_ql, ql_pol = ql_policies[p]
        ql_stats = rollout_stats(world_ql, ql_pol)
        out[f"{p:.2f}"] = {"VI": vi_stats, "QL": ql_stats}
        print(f"p={p:.2f}")
        print("  VI:", vi_stats)
        print("  QL:", ql_stats)
    path = os.path.join(RES_DIR, "route_analysis.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("saved", path)


if __name__ == "__main__":
    main()
