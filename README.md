# COSC-4117EL Assignment 2 — Group 1
## Warehouse Robot Decision Making: Value Iteration vs Q-Learning

This folder contains the complete, runnable submission for the warehouse
robot MDP / RL assignment. The grid layout, rewards, station placement and
pygame GUI of the provided `warehouse_gridworld_domain_random.py` are
unchanged; the environment was extended **only** to support the stochastic
motion model required by the brief (see the change log in the file header).

## Files

| File | Purpose |
|---|---|
| `warehouse_gridworld_domain_random.py` | Provided environment, extended with `intended_prob` (stochastic transitions), `sample_realized_action()` and a pure `transition_kernel()` used by both the simulator and Value Iteration. |
| `COSC_4117EL_A2_G1-value_iteration.py` | **Part 1 + Part 3 (DP side).** Builds the known MDP, runs Value Iteration for p = 1.00 / 0.90 / 0.80, extracts policies, evaluates them, and saves policy figures, the convergence figure and `results/vi_results.json`. |
| `COSC_4117EL_A2_G1-q_learning.py` | **Part 2 + Part 3 (RL side).** Tabular Q-Learning with epsilon-greedy exploration and geometric epsilon decay; two hyperparameter configurations over three seeds; greedy evaluation; Part 3 comparison against the Value Iteration results; learning curves, policy and trajectory figures; `results/ql_results.json`. |
| `route_analysis.py` | Supplementary Part 3 analysis: counts congestion-zone entries, blocked moves and path lengths for each method and p (`results/route_analysis.json`). |
| `make_report.py` | Regenerates the PDF report (`COSC_4117EL_A2_G1_Report.pdf`) from the JSON results and figures. |
| `COSC_4117EL_A2_G1_Report.pdf` | The report (MDP formulation, implementation, experiments, comparison, analysis). |
| `figures/` | All policy/learning/comparison figures used in the report. |
| `results/` | Machine-readable result tables (JSON). |

## Requirements

- Python 3.10+
- `numpy`, `matplotlib`, `pygame` (the community build `pygame-ce` also works
  and provides the `pygame` package). The report generator additionally needs
  `reportlab`.

```bash
pip install numpy matplotlib pygame reportlab
```

No virtual environments or installed packages are included in this ZIP.

## How to run

Run the commands from inside this folder. **No manual editing is required**;
all seeds, layout (seed 99, matching Figure 1) and paths are fixed in the
scripts.

```bash
# 1. Part 1 + Part 3 (Value Iteration) — fast, a few seconds
python COSC_4117EL_A2_G1-value_iteration.py

# 2. Part 2 + Part 3 (Q-Learning) — a few minutes (training, 2 configs x
#    3 seeds x 8000 episodes plus 3 p x 3 seeds x 6000 episodes).
#    It automatically runs the Value Iteration script first if results/vi_results.json is missing.
python COSC_4117EL_A2_G1-q_learning.py

# 3. Optional: route statistics for the Part 3 discussion
python route_analysis.py

# 4. Optional: rebuild the PDF report from the generated results/figures
python make_report.py

# Optional: the original manual GUI (arrow keys, stochastic probability defaults to 1.0)
python warehouse_gridworld_domain_random.py
```

All figures are written to `figures/` and all numerical results to
`results/`. The scripts are headless (`matplotlib` Agg backend) and never
open the pygame window during training.

## Fixed experimental settings

- **Layout seed:** 99 (Pickup P = (8, 0), Packing D = (7, 7), Charging dock C = (4, 2)).
- **Discount factor:** gamma = 0.95 (Value Iteration and Q-Learning config A).
- **VI convergence threshold:** theta = 1e-6 on the maximum Bellman residual.
- **Q-Learning training seeds:** 42, 7, 123 (agent RNG; environment slip RNG is
  re-seeded independently).
- **Evaluation:** 500 greedy episodes (exploration disabled) per trained policy,
  using identical evaluation RNG seeds for both methods.
- **Stochastic motion:** intended action with probability p, each perpendicular
  direction with (1-p)/2; a blocked realised move leaves the robot in place with
  the -3 invalid-move penalty (plus the -1 step penalty).

## Reproducibility notes

- Deterministic with respect to the listed seeds (NumPy `default_rng`).
- Value Iteration is deterministic; the only randomness is in Monte Carlo
  evaluation roll-outs, which use fixed seeds.
- Q-Learning results vary slightly across machines only if NumPy version /
  RNG stream differs; conclusions are identical across all tested seeds.

## Generative AI

Generative AI (Doubao, ByteDance) was used to draft and debug the code and the
report; the initial environment template itself notes ChatGPT assistance.
All AI use is acknowledged in the report, including the prompts used. Please
add the corresponding share links before submission (see the report's
"Generative AI use statement" section).
