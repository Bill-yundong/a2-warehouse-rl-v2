# -*- coding: utf-8 -*-
"""
Builds the Assignment 2 PDF report (COSC_4117EL_A2_G1_Report.pdf) from the
JSON results and figures produced by the two algorithm scripts.

Usage:
    python make_report.py
"""

import json
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from warehouse_gridworld_domain_random import WarehouseGridWorld
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
RES = os.path.join(HERE, "results")

_spec = importlib.util.spec_from_file_location(
    "vi", os.path.join(HERE, "COSC_4117EL_A2_G1-value_iteration.py"))
vi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vi)

with open(os.path.join(RES, "vi_results.json"), encoding="utf-8") as f:
    VI = json.load(f)
with open(os.path.join(RES, "ql_results.json"), encoding="utf-8") as f:
    QL = json.load(f)
with open(os.path.join(RES, "route_analysis.json"), encoding="utf-8") as f:
    ROUTE = json.load(f)

# Derived MDP counts.
_world = WarehouseGridWorld(seed=99, intended_prob=0.8)
_states = vi.enumerate_states(_world)
_n_cells = sum(1 for r in range(10) for c in range(10) if _world.grid[r, c] != 1)
_n_states = len(_states)
_n_terminal = sum(1 for s in _states if vi.is_terminal(_world, s))

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="H1", parent=styles["Heading1"],
                          fontSize=15, spaceBefore=14, spaceAfter=6,
                          textColor=colors.HexColor("#1a3a6b")))
styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"],
                          fontSize=12.5, spaceBefore=10, spaceAfter=4,
                          textColor=colors.HexColor("#244a7e")))
styles.add(ParagraphStyle(name="Body", parent=styles["BodyText"],
                          fontSize=10, leading=14, alignment=TA_JUSTIFY,
                          spaceAfter=5))
styles.add(ParagraphStyle(name="BulletItem", parent=styles["BodyText"],
                          fontSize=10, leading=13.5, leftIndent=14,
                          spaceAfter=2))
styles.add(ParagraphStyle(name="CapItem", parent=styles["BodyText"],
                          fontSize=8.8, leading=11, alignment=TA_CENTER,
                          textColor=colors.HexColor("#444444"),
                          spaceAfter=10))
styles.add(ParagraphStyle(name="TitleBigItem", parent=styles["Title"],
                          fontSize=20, leading=24))
styles.add(ParagraphStyle(name="Sub", parent=styles["Normal"],
                          fontSize=11, alignment=TA_CENTER, leading=16))

story = []


def H1(t): story.append(Paragraph(t, styles["H1"]))
def H2(t): story.append(Paragraph(t, styles["H2"]))
def P(t): story.append(Paragraph(t, styles["Body"]))
def B(t): story.append(Paragraph("&bull;&nbsp;&nbsp;" + t, styles["BulletItem"]))
def C(t): story.append(Spacer(1, 3))
def fig(name, width=16.4 * cm, caption=None):
    path = os.path.join(FIG, name)
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        w, h = im.size
    story.append(Image(path, width=width, height=width * h / w))
    if caption:
        story.append(Paragraph(caption, styles["CapItem"]))


def table(data, col_widths=None, header=True, font_size=9):
    t = Table(data, colWidths=col_widths, hAlign="CENTER")
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#888888")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#244a7e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#eef3fa")]),
        ]
    t.setStyle(TableStyle(style))
    story.append(t)
    story.append(Spacer(1, 8))


# ---------------------------------------------------------------------------
# Title page
# ---------------------------------------------------------------------------
story.append(Spacer(1, 3.5 * cm))
story.append(Paragraph("COSC-4117EL Assignment 2", styles["TitleBigItem"]))
story.append(Spacer(1, 0.4 * cm))
story.append(Paragraph(
    "Warehouse Robot Decision Making with<br/>Dynamic Programming and "
    "Reinforcement Learning", styles["Sub"]))
story.append(Spacer(1, 1.2 * cm))
story.append(Paragraph("Group 1 (G1)", styles["Sub"]))
story.append(Paragraph("Group members: [fill in names / student IDs]", styles["Sub"]))
story.append(Spacer(1, 0.8 * cm))
story.append(Paragraph(
    "Fixed layout: seed 99 (Pickup P = (8, 0), Packing D = (7, 7), "
    "Charging dock C = (4, 2))<br/>Discount factor &gamma; = 0.95",
    styles["Sub"]))
story.append(PageBreak())

# ---------------------------------------------------------------------------
# 1. Introduction
# ---------------------------------------------------------------------------
H1("1. Introduction and MDP formulation (Part 1, Q1)")
P("A warehouse robot starts on the charging dock (green cell C), must first "
  "collect a package from the pickup station (blue P), deliver it to the "
  "packing station (yellow D) while carrying it, and finally return to the "
  "charging dock. Black shelves block movement and red congestion zones can "
  "be entered but incur a penalty. The experiments use the fixed seeded "
  "layout (seed 99) shown in Figure 1 of the brief; the provided environment "
  "was only extended to implement the stochastic motion model (documented in "
  "the file header).")

H2("1.1 State space S")
P("The state is the tuple <b>s = (row, col, has_package, delivered)</b>, "
  "where row, col &isin; {0,...,9} and the two task flags are binary. "
  "Position (x, y) alone is <b>not</b> sufficient: the reward received on the "
  "packing cell and on the dock and the set of optimal actions depend on the "
  "task phase. For example, standing on the dock with no package is an "
  "ordinary state, whereas standing on the dock after delivery is terminal. "
  "Adding the two binary flags makes the future depend only on the current "
  "state (the Markov property). The combination (has_package=0, delivered=1) "
  "is logically impossible and is excluded. Shelf cells are excluded as well. "
  f"For the seed-99 layout there are {_n_cells} passable cells, giving "
  f"{_n_states} reachable MDP states (three task phases each), of which "
  f"{_n_terminal} is the success terminal state.")

H2("1.2 Action space A")
P("A = {up, down, left, right}. All four actions are available everywhere; "
  "a move that would enter a shelf or leave the grid leaves the robot in the "
  "same cell and incurs the invalid-move penalty (rather than being removed "
  "from the action set).")

H2("1.3 Transition model P(s'|s,a)")
P("When the robot selects an intended action a, the action occurs with "
  "probability p; the remaining probability 1-p is split equally between the "
  "two perpendicular directions (no backwards move). For Parts 1-2, p = 0.80, "
  "so e.g. choosing up yields up with 0.8, left with 0.1 and right with 0.1. "
  "If a realised movement would enter a shelf or leave the grid, the robot "
  "stays in place. Part 3 also evaluates p = 0.90 and p = 1.00.")

H2("1.4 Reward model R(s,a,s')")
table([
    ["Event", "Reward"],
    ["Pick up the package at the pickup station", "+25"],
    ["Deliver the package at the packing station while carrying", "+40"],
    ["Return to the charging dock after delivery", "+100"],
    ["Enter a congestion zone (red cell)", "-10"],
    ["Every time step", "-1"],
    ["Attempted move blocked by a shelf / grid edge (realised)", "-3"],
], col_widths=[11.5 * cm, 3 * cm])
P("Rewards are additive; for example a realised blocked step costs -1 + (-3) "
  "= -4, and a blocked step that lands on a congestion cell is impossible by "
  "construction (the robot stays put).")

H2("1.5 Discount factor and terminal conditions")
P("The discount factor is <b>&gamma; = 0.95</b>, which keeps the task "
  "valuable (all goal rewards can be earned in tens of steps) while still "
  "preferring shorter routes. The <b>success terminal</b> condition is being "
  "on the charging dock with delivered = 1 (the +100 reward is paid and the "
  "episode ends). The 120-step limit is a simulation horizon (an episode "
  "ending with 'max_steps' is counted as a failure), not a stationary MDP "
  "state; Value Iteration therefore models only the success terminal, while "
  "the simulator applies the step limit.")

# ---------------------------------------------------------------------------
# 2. Value Iteration
# ---------------------------------------------------------------------------
H1("2. Value Iteration (Part 1, Q2)")
P("The known transition and reward models are enumerated directly from the "
  "environment's pure transition kernel (the same function the simulator "
  "uses, so the planner's model cannot disagree with the environment). Value "
  "Iteration repeatedly applies the Bellman optimality update "
  "V<sub>k+1</sub>(s) = max<sub>a</sub> &sum;<sub>s'</sub> P(s'|s,a) "
  "[R(s,a,s') + &gamma; V<sub>k</sub>(s')] over all non-terminal states, "
  "starting from V<sub>0</sub> = 0, until "
  "max<sub>s</sub>|V<sub>k+1</sub>(s) - V<sub>k</sub>(s)| &lt; "
  "&theta;, with <b>&theta; = 10<sup>-6</sup></b>. The greedy policy is then "
  "extracted as argmax<sub>a</sub> of the Bellman backup.")
r08 = VI["runs"]["0.80"]; r09 = VI["runs"]["0.90"]; r10 = VI["runs"]["1.00"]
table([
    ["Intended-action p", "Sweeps to converge", "Final max residual",
     "V at start state"],
    ["1.00", str(r10["iterations_to_converge"]), "0 (exact)",
     f"{r10['value_at_start']:.2f}"],
    ["0.90", str(r09["iterations_to_converge"]),
     f"{r09['final_max_delta']:.2e}", f"{r09['value_at_start']:.2f}"],
    ["0.80", str(r08["iterations_to_converge"]),
     f"{r08['final_max_delta']:.2e}", f"{r08['value_at_start']:.2f}"],
], col_widths=[3.4 * cm, 4.0 * cm, 4.0 * cm, 3.6 * cm])
P(f"For the Part 1 setting (p = 0.80) Value Iteration converged in "
  f"<b>{r08['iterations_to_converge']} sweeps</b>; convergence is slower as "
  f"motion becomes noisier (38 sweeps at p = 1.00, 68 at p = 0.90, 86 at "
  f"p = 0.80), because stochastic transitions spread value information over "
  f"more neighbouring states per backup.")
fig("vi_convergence.png", 14.5 * cm,
    "Figure 1. Maximum Bellman residual per sweep (log scale). "
    "More stochastic motion needs more sweeps to reach theta = 1e-6.")
fig("vi_policy_p080.png", 17.0 * cm,
    "Figure 2. Value Iteration policy for p = 0.80 in the three task phases: "
    "go to pickup (left), carry to packing (middle), return to dock (right). "
    "The magenta square marks a state whose preferred action changes as p "
    "decreases (analysed in Section 4.4).")
P("The policy is intuitively correct: in phase 1 it routes down the western "
  "edge to P, in phase 2 it crosses the bottom corridor and comes up column "
  "6/7 to D, and in phase 3 it returns along row 4/5 to C. It deliberately "
  "uses one congestion cell ((3,5)) on the return leg because the deterministic "
  "shortest detour around it costs more than the -10 penalty.")
P("<b>Why a known model is required.</b> Every Bellman backup explicitly sums "
  "over all possible successor states using P(s'|s,a) and R(s,a,s'). Without "
  "the model the expected values of untried actions could not be evaluated "
  "without sampling them, which is exactly why Part 2 needs a model-free "
  "learning algorithm.")
story.append(PageBreak())

# ---------------------------------------------------------------------------
# 3. Q-Learning
# ---------------------------------------------------------------------------
H1("3. Q-Learning (Part 2)")
P("A Q-table of size 400 x 4 (state index x four actions) is initialised to "
  "zero. The agent acts epsilon-greedy, with epsilon decayed geometrically "
  "each episode from 1.0 to a floor of 0.01, and applies the standard update "
  "Q(s,a) &larr; Q(s,a) + &alpha;[r + &gamma;(1-done)max<sub>a'</sub>"
  "Q(s',a') - Q(s,a)]. The agent never accesses P(s'|s,a); it learns only "
  "from sampled experience. Two hyperparameter configurations were compared "
  "over three training seeds (42, 7, 123), using the fixed seed-99 layout:")
a = QL["part2"]["configs"]["A_conservative"]["cfg"]
b = QL["part2"]["configs"]["B_fast_adapting"]["cfg"]
table([
    ["Config", "alpha", "gamma", "eps decay", "eps range", "episodes"],
    ["A (conservative)", str(a["alpha"]), str(a["gamma"]),
     str(a["eps_decay"]), "1.00 -> 0.01", str(a["episodes"])],
    ["B (fast adapting)", str(b["alpha"]), str(b["gamma"]),
     str(b["eps_decay"]), "1.00 -> 0.01", str(b["episodes"])],
], col_widths=[3.6 * cm, 1.8 * cm, 1.8 * cm, 2.4 * cm, 2.8 * cm, 2.2 * cm])
fig("ql_learning_curves.png", 14.8 * cm,
    "Figure 3. Training curves at p = 0.80 (100-episode rolling mean; shaded "
    "band = +/- standard deviation across seeds 42, 7, 123). Top: episode "
    "return; bottom: task success rate.")
P("Both configurations learn the task: the rolling success rate approaches "
  "1.0 after roughly 1.3k-2k episodes and the return approaches the Value "
  "Iteration level. Config B (large alpha 0.5, high gamma 0.99, faster "
  "exploration decay) learns faster early but is noisier and less robust: "
  "for seed 123 its fully greedy policy becomes brittle (a 1%-greedy "
  "exploration rate during training masks cycles that pure greedy execution "
  "then falls into), giving an evaluation success rate of only 0.50. Config A "
  "(smaller alpha, slower decay) is smooth and succeeds on all three seeds.")

A_eval = QL["part2"]["configs"]["A_conservative"]["eval"]
B_eval = QL["part2"]["configs"]["B_fast_adapting"]["eval"]
rows = [["Config", "seed", "avg return", "success rate", "avg steps"]]
for cfg_eval, label in ((A_eval, "A"), (B_eval, "B")):
    for ps in cfg_eval["per_seed"]:
        rows.append([label, str(ps["seed"]), f"{ps['avg_return']:.2f}",
                     f"{ps['success_rate']:.3f}", f"{ps['avg_steps']:.1f}"])
    rows.append([label, "mean +/- std",
                 f"{cfg_eval["avg_return_mean"]:.2f} +/- {cfg_eval['avg_return_std']:.2f}",
                 f"{cfg_eval['success_rate_mean']:.3f}",
                 f"{cfg_eval['avg_steps_mean']:.1f}"])
table(rows, col_widths=[2.4 * cm, 3.0 * cm, 4.0 * cm, 3.0 * cm, 2.6 * cm])
P("Config A is therefore chosen for the Part 3 comparison. After training "
  "the greedy policy was extracted and evaluated with exploration disabled "
  "(500 episodes, identical evaluation seeds for both methods). Figure 4 "
  "shows the resulting policy and Figure 5 one successful greedy run (the "
  "orange rings mark the pickup and delivery events).")
fig("ql_policy_p080.png", 17.0 * cm,
    "Figure 4. Q-Learning greedy policy (config A, best seed 42) at p = 0.80.")
fig("ql_demo_trajectory_p080.png", 10.5 * cm,
    "Figure 5. One successful greedy evaluation run: dock -> pickup -> "
    "packing -> dock in 27 steps at p = 0.80.")

H2("3.1 Comparison with the Value Iteration policy")
P("At p = 0.80 both methods reach the same performance (Section 4), but the "
  "Q-Learning and Value Iteration arrows agree on only about 72% of all "
  "grid cells. Restricting the comparison to states the trained agent "
  "actually visits during greedy evaluation, agreement rises to about 94% "
  "(100% at p = 1.00). The remaining whole-grid differences are harmless: "
  "Q-Learning rarely visits states far from the successful corridors, so "
  "their Q-values stay near zero and argmax picks arbitrary actions, and "
  "some near-tied routes (equal expected return) are resolved differently. "
  "The two methods learn the same behaviour where the behaviour matters.")
story.append(PageBreak())

# ---------------------------------------------------------------------------
# 4. Part 3
# ---------------------------------------------------------------------------
H1("4. Stochastic environment, comparison and visualisation (Part 3)")
H2("4.1 Quantitative comparison under p = 1.00 / 0.90 / 0.80")
rows = [["p", "VI sweeps", "VI return", "QL return (3 seeds)",
         "success VI / QL", "VI steps", "QL steps",
         "QL converge episode*"]]
for r in QL["part3"]["comparison"]:
    p = f"{r['p']:.2f}"
    vi_r = VI["runs"][p]
    rows.append([
        p, str(vi_r["iterations_to_converge"]),
        f"{vi_r['avg_return']:.2f}",
        f"{r['ql_avg_return_mean']:.2f} +/- {r['ql_avg_return_std']:.2f}",
        f"{vi_r['success_rate']:.3f} / {r['ql_success_rate_mean']:.3f}",
        f"{vi_r['avg_steps']:.1f}", f"{r['ql_avg_steps_mean']:.1f}",
        str(r["ql_convergence_episode_median"])])
table(rows, col_widths=[1.3 * cm, 1.7 * cm, 1.9 * cm, 2.9 * cm, 2.7 * cm,
                        1.7 * cm, 1.7 * cm, 2.3 * cm], font_size=8)
P("*Median training seed; first episode after which the 100-episode rolling "
  "success rate stays at least 0.9.")
fig("part3_comparison.png", 15.5 * cm,
    "Figure 6. Average cumulative return and task success rate over 500 "
    "greedy evaluation episodes for both methods.")
P("Q-Learning matches the model-based optimum almost exactly at every p "
  "(differences below 0.02 reward). Both methods always complete the task, "
  "but average return drops from 131 to 101 and path length grows from 24 to "
  "about 32 steps as p decreases: noise causes slips, blocked moves and "
  "extra congestion entries. Value Iteration needs more Bellman sweeps under "
  "noise, while Q-Learning needs more interaction episodes (about 900 at "
  "p = 1.00 vs about 1280 at p = 0.80).")

H2("4.2 Route behaviour near congestion zones")
rows = [["p", "mean congestion entries (VI / QL)",
         "mean blocked moves (VI / QL)", "mean path length (VI / QL)"]]
for p in ("1.00", "0.90", "0.80"):
    rows.append([
        p,
        f"{ROUTE[p]['VI']['mean_congestion_entries']:.2f} / "
        f"{ROUTE[p]['QL']['mean_congestion_entries']:.2f}",
        f"{ROUTE[p]['VI']['mean_blocked_moves']:.2f} / "
        f"{ROUTE[p]['QL']['mean_blocked_moves']:.2f}",
        f"{ROUTE[p]['VI']['mean_length']:.1f} / "
        f"{ROUTE[p]['QL']['mean_length']:.1f}"])
table(rows, col_widths=[1.6 * cm, 5.4 * cm, 4.6 * cm, 4.6 * cm])
P("Two effects are visible. First, the <b>optimal policy changes</b> to avoid "
  "corridors that are risky under slipping (Section 4.4) - for example, "
  "preferring the open eastern column over the shelf-pinched western side "
  "when in the top-right congestion cluster. Second, even with rerouting, "
  "random slips still push the robot into red cells, so realised congestion "
  "entries (1.0 -> 1.44 -> 2.06) and blocked moves (0 -> 1.6 -> 3.6) "
  "increase with uncertainty. The learned Q-Learning policies reproduce the "
  "same routing behaviour and statistics.")
fig("vi_policy_p100.png", 17.0 * cm,
    "Figure 7. Value Iteration policy at p = 1.00 (deterministic).")
fig("vi_policy_p090.png", 17.0 * cm,
    "Figure 8. Value Iteration policy at p = 0.90.")

H2("4.3 States whose preferred action changes")
n_ch = VI["n_policy_changes"]
n_ch_c = VI["n_policy_changes_near_congestion"]
P(f"Comparing the p = 1.00 and p = 0.80 Value Iteration policies, "
  f"<b>{n_ch} states</b> change their preferred action, "
  f"{n_ch_c} of which are on or immediately adjacent to a congestion cell. "
  "Examples include (0,8,0,0) and (2,8,0,0) changing left -> right, "
  "(2,5,0,0) changing down -> up, and (5,1,0,0) changing down -> left.")

H2("4.4 Worked example: state (0, 8, 0, 0)")
P("This state sits on the congestion cell in the top-right corner during "
  "phase 1. The one-step outcomes of moving left or right are symmetric "
  "(0.8 step to the neighbouring cell, 0.1 blocked slip at the top edge, "
  "0.1 slip down into congestion (1,8)), so the difference is driven by the "
  "value of the continuation:")
table([
    ["p", "Q(left)", "Q(right)", "preferred", "V(0,7)", "V(0,9)"],
    ["1.00", "14.46", "14.37", "left", "16.27", "16.18"],
    ["0.90", "1.26", "1.62", "right", "-", "-"],
    ["0.80", "-12.22", "-11.79", "right", "-9.41", "-8.85"],
], col_widths=[2.0 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm])
P("At p = 1.00 the western continuation is marginally better. At p = 0.80 "
  "the order flips: column 9 is an open straight corridor (cells (1,9), "
  "(2,9), (3,9) are all empty, so a slip rarely causes a penalty), whereas "
  "the western route through column 7 is squeezed between shelves at "
  "(0,6)/(1,6) and the row-3 shelf block (columns 6-8 are shelves), forcing "
  "a left turn while a downward slip can be blocked (-3) or bounce the robot "
  "back towards the congestion cell. With 20% slip probability the safer "
  "eastern continuation wins, so the preferred action changes from left to "
  "right. This is exactly the risk-averse rerouting the Bellman backup "
  "produces when the transition model is stochastic.")
story.append(PageBreak())

# ---------------------------------------------------------------------------
# 5. Conclusion
# ---------------------------------------------------------------------------
H1("5. Conclusion")
B("The warehouse task is an MDP with state (row, col, has_package, "
  "delivered); position alone violates the Markov property.")
B("Value Iteration with gamma = 0.95 and theta = 1e-6 converged in 86 "
  "sweeps at p = 0.80 and yields the optimal policy directly from the known "
  "model.")
B("Model-free Q-Learning (alpha = 0.1, gamma = 0.95, epsilon decayed "
  "1.0 -> 0.01) reaches the same performance over three seeds, with "
  "learning converging around 1.3k-2k episodes; a more aggressive "
  "configuration learns faster but is less robust to seed effects.")
B("Under p = 1.00 / 0.90 / 0.80 both methods achieve a 100% success rate, "
  "with average return 131.00 / 118.36 / 101.36 and mean path length "
  "24.0 / 27.5 / 32.1. QL and VI agree on essentially all states the "
  "learned policy actually visits.")
B("Increasing uncertainty changes the preferred route near congestion and "
  "wall pinch points: 15 states flip actions, including (0,8,0,0) where the "
  "policy switches from the shelf-pinched western side to the open eastern "
  "corridor.")

H1("6. Generative AI use statement")
P("The original environment template header states that it was generated "
  "with the assistance of ChatGPT. For this submission, the Doubao "
  "(ByteDance) generative AI assistant was used to: (i) extend the "
  "environment with the stochastic transition kernel; (ii) draft and debug "
  "the Value Iteration and Q-Learning implementations and the plotting "
  "code; (iii) draft the report text. All code was reviewed, run and "
  "verified by the group, and every numerical result in this report was "
  "produced by the submitted scripts. The following prompts were used "
  "(please paste the share links generated by the tool before submission):")
B("Implement stochastic transitions (intended action p, perpendicular "
  "slips (1-p)/2) and a pure transition kernel for the provided warehouse "
  "gridworld, and explain the MDP state representation.")
B("Implement Value Iteration with the Bellman optimality update, "
  "convergence threshold, greedy policy extraction and arrow "
  "visualisations for the three task phases.")
B("Implement tabular Q-Learning with epsilon-greedy decay, two alpha/gamma "
  "configurations, multiple seeds, learning curves and greedy evaluation; "
  "compare it with the Value Iteration results under p = 1.00/0.90/0.80.")
B("Explain why the preferred action changes at state (0,8,0,0) as p "
  "decreases, and draft the report sections.")

H1("7. Group member contribution statement")
P("[Fill in: contributions of each member. All members confirm that the "
  "work is submitted in accordance with the Student Academic Integrity "
  "policy. No grade adjustment is requested / if one is requested, describe "
  "the situation and attach the agreement of all members.]")

# ---------------------------------------------------------------------------
doc = SimpleDocTemplate(
    os.path.join(HERE, "COSC_4117EL_A2_G1_Report.pdf"),
    pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
    topMargin=1.8 * cm, bottomMargin=1.8 * cm,
    title="COSC-4117EL Assignment 2 Report (G1)")
doc.build(story)
print("Report written:", os.path.join(HERE, "COSC_4117EL_A2_G1_Report.pdf"))


