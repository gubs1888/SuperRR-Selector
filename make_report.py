#!/usr/bin/env python3
"""Build charts, the methodology report (Markdown/HTML) and a PDF."""

import csv
import json
import os
import re
import shutil
import subprocess
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import markdown
from bs4 import BeautifulSoup

import pipeline

OUT = "outputs"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "Carlito",
    "font.size": 9.5,
    "axes.edgecolor": "#cbd5e1",
    "axes.labelcolor": "#334155",
    "text.color": "#1e293b",
    "axes.titlecolor": "#0b1f3a",
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.grid": False,
    "grid.color": "#e2e8f0",
    "grid.linewidth": 0.7,
    "legend.frameon": False,
})


def load(name):
    with open(os.path.join(OUT, name), newline="") as fh:
        return list(csv.DictReader(fh))


bat = load("player_pressure_ratings_bat.csv")
bowl = load("player_pressure_ratings_bowl.csv")
summary = json.load(open(os.path.join(OUT, "summary.json")))
balls = load("ball_pressure.csv")
meta = pipeline.build_match_meta()

thr_bat = summary["pressure_thresholds"]["bat"]
thr_bowl = summary["pressure_thresholds"]["bowl"]

# ---------------------------------------------------------------------------
# Extra statistics
# ---------------------------------------------------------------------------
faced = [r for r in balls if r["faced"] == "1"]
legal = [r for r in balls if r["legal"] == "1"]
pi_bat = np.array([float(r["pi_bat"]) for r in faced])
pi_bowl = np.array([float(r["pi_bowl"]) for r in legal])

tier_names = {
    "bilateral": "Bilateral",
    "qualifier": "Qualifier",
    "tri_series": "Tri-series",
    "emerging_league": "Emerging league",
    "emerging_knockout": "Emerging KO",
    "ipl_league": "IPL league",
    "ipl_playoff": "IPL playoffs",
    "continental_league": "Asia Cup league",
    "continental_knockout": "Asia Cup final",
}
tier_counts = Counter(m["tier"] for m in meta.values())
season_matches = Counter(m["season"] for m in meta.values())
season_balls = Counter()
for r in balls:
    m = meta.get(r["match_id"])
    if m:
        season_balls[m["season"]] += 1
tier_pi = defaultdict(lambda: {"bat": [], "bowl": []})
tier_pressure = Counter()
for r in balls:
    m = meta.get(r["match_id"])
    if not m:
        continue
    t = m["tier"]
    if r["faced"] == "1":
        tier_pi[t]["bat"].append(float(r["pi_bat"]))
        if float(r["pi_bat"]) >= thr_bat:
            tier_pressure[t] += 1
    if r["legal"] == "1":
        tier_pi[t]["bowl"].append(float(r["pi_bowl"]))
        if float(r["pi_bowl"]) >= thr_bowl:
            tier_pressure[t] += 1

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def fig_distribution():
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    bins = np.linspace(0, 140, 57)
    ax.hist(pi_bat, bins=bins, alpha=0.68, label="Batting deliveries", color="#0f766e", edgecolor="none")
    ax.hist(pi_bowl, bins=bins, alpha=0.58, label="Bowling deliveries", color="#d97706", edgecolor="none")
    ax.axvline(thr_bat, color="#0b1f3a", ls="--", lw=1.1)
    ax.axvline(thr_bowl, color="#b45309", ls="--", lw=1.1)
    ax.text(thr_bat + 1, ax.get_ylim()[1] * 0.92, f"bat threshold {thr_bat:.1f}", color="#0b1f3a", fontsize=8)
    ax.text(thr_bowl + 1, ax.get_ylim()[1] * 0.80, f"bowl threshold {thr_bowl:.1f}", color="#b45309", fontsize=8)
    ax.set_xlabel("Pressure Index (PI)")
    ax.set_ylabel("Deliveries")
    ax.set_title("Distribution of the Pressure Index")
    ax.legend(loc="upper right")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_pressure_dist.png"), dpi=200)
    plt.close(fig)


def fig_dumbbell(rows, kind):
    rows = rows[:10]
    labels = [f"{r['rank']}. {r['player_name']}" for r in rows]
    if kind == "bat":
        base = [float(r["sr_baseline"]) for r in rows]
        press = [float(r["sr_pressure_actual"]) for r in rows]
        xlabel = "Strike rate"
        c = "#0f766e"
        title = "Top 10 batters: usual vs pressure strike rate"
    else:
        base = [float(r["economy_baseline"]) for r in rows]
        press = [float(r["economy_pressure_actual"]) for r in rows]
        xlabel = "Economy rate (lower is better)"
        c = "#d97706"
        title = "Top 10 bowlers: usual vs pressure economy"
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    for yi, b, p in zip(y, base, press):
        ax.plot([b, p], [yi, yi], color="#cbd5e1", lw=2.2, zorder=1)
    ax.scatter(base, y, color="#94a3b8", s=38, label="Usual", zorder=2)
    ax.scatter(press, y, color=c, s=50, label="Under pressure", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.2)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"fig_top_{kind}.png"), dpi=200)
    plt.close(fig)


def fig_tiers():
    order = ["bilateral", "qualifier", "emerging_league", "tri_series", "continental_league",
             "emerging_knockout", "continental_knockout", "ipl_league", "ipl_playoff"]
    labels = [tier_names[t] for t in order]
    counts = [tier_counts[t] for t in order]
    mean_bat = [np.mean(tier_pi[t]["bat"]) if tier_pi[t]["bat"] else 0 for t in order]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    axes[0].barh(labels, counts, color="#0b1f3a", height=0.68)
    axes[0].set_title("Matches by competition tier")
    axes[0].set_xlabel("Matches")
    for i, v in enumerate(counts):
        axes[0].text(v + 3, i, str(v), va="center", fontsize=8.2, color="#475569")
    axes[0].grid(axis="x")
    axes[0].set_axisbelow(True)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].tick_params(axis="y", length=0)

    axes[1].barh(labels, mean_bat, color="#0f766e", height=0.68)
    axes[1].set_title("Mean batting Pressure Index")
    axes[1].set_xlabel("Mean PI")
    for i, v in enumerate(mean_bat):
        axes[1].text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=8.2, color="#475569")
    axes[1].grid(axis="x")
    axes[1].set_axisbelow(True)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].tick_params(axis="y", length=0)
    fig.tight_layout(pad=1.2)
    fig.savefig(os.path.join(FIG, "fig_tiers.png"), dpi=200)
    plt.close(fig)


fig_distribution()
fig_dumbbell(bat, "bat")
fig_dumbbell(bowl, "bowl")
fig_tiers()

# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def bat_table(rows):
    lines = ["| Rank | Player | Team | Pressure balls | SR usual | SR pressure | SR par | Final score | Consistency |",
             "|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(
            f"| {r['rank']} | {r['player_name']} | {r['team']} | {r['pressure_balls']} | "
            f"{r['sr_baseline']} | {r['sr_pressure_actual']} | {r['expected_sr']} | "
            f"{float(r['final_score']):+.3f} | {r['consistency']} |")
    return "\n".join(lines)


def bowl_table(rows):
    lines = ["| Rank | Player | Team | Pressure balls | Econ usual | Econ pressure | Econ par | Wkts | Final score | Consistency |",
             "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(
            f"| {r['rank']} | {r['player_name']} | {r['team']} | {r['pressure_balls']} | "
            f"{r['economy_baseline']} | {r['economy_pressure_actual']} | {r['expected_economy']} | "
            f"{r['pressure_wickets']} | {float(r['final_score']):+.3f} | {r['consistency']} |")
    return "\n".join(lines)


def season_table():
    lines = ["| Season | Matches | Deliveries |", "|---|---:|---:|"]
    for s in sorted(season_matches):
        lines.append(f"| {s} | {season_matches[s]} | {season_balls[s]:,} |")
    return "\n".join(lines)


def tier_mean(tier, kind):
    vals = tier_pi[tier][kind]
    return float(np.mean(vals)) if vals else 0.0


def tier_share(tier):
    total = len(tier_pi[tier]["bat"]) + len(tier_pi[tier]["bowl"])
    return tier_pressure[tier] / total if total else 0.0


def tier_table():
    order = ["bilateral", "qualifier", "emerging_league", "tri_series",
             "continental_league", "emerging_knockout", "continental_knockout",
             "ipl_league", "ipl_playoff"]
    lines = ["| Tier | Matches | Batting balls | Mean batting PI | Mean bowling PI | Pressure share |",
             "|---|---:|---:|---:|---:|---:|"]
    for t in order:
        lines.append(
            f"| {tier_names[t]} | {tier_counts[t]} | {len(tier_pi[t]['bat']):,} | "
            f"{tier_mean(t, 'bat'):.1f} | {tier_mean(t, 'bowl'):.1f} | {tier_share(t):.0%} |")
    return "\n".join(lines)


def headline_table():
    lines = ["| Rank | Batter | Score | Bowler | Score |",
             "|---:|---|---:|---|---:|"]
    for i in range(5):
        lines.append(
            f"| {i + 1} | {bat[i]['player_name']} | {float(bat[i]['final_score']):+.3f} | "
            f"{bowl[i]['player_name']} | {float(bowl[i]['final_score']):+.3f} |")
    return "\n".join(lines)


cfg = summary["config"]
phase = summary["phase_par"]
wp = summary["win_probability_models"]
wp1 = wp["1"]["weights"]
wp2 = wp["2"]["weights"]
def bat_idx(name):
    return next(i for i, r in enumerate(bat) if r["player_name"] == name)


def bowl_idx(name):
    return next(i for i, r in enumerate(bowl) if r["player_name"] == name)


bum_idx = bowl_idx("Jasprit Bumrah")

report = f"""# SupeRR Selector Season 2 — Round 1

## Identifying the most reliable players under pressure

**Submission type:** data-driven player rankings + methodology report  
**Data:** IPL 2022–2026 and T20 Internationals 2022–2026 supplied in the Round 1 data pack  
**Coverage:** {summary['matches']} matches, {summary['deliveries']:,} deliveries, 2022–2026

---

## 1. Executive summary

We build a **Pressure Index (PI)** for every delivery from three ingredients:

1. **Match stakes** — a competition/stage weight that follows the requested ladder:
   bilateral and Asia Cup group games < IPL league < IPL playoffs, with the Asia Cup final
   deliberately kept below IPL league level rather than treated as a World Cup stand-in.
2. **Opponent strength** — the opponent's pre-match Elo rating, used as a dynamic proxy for
   the ICC T20I ranking because the supplied pack contains no ranking file.
3. **Situational pressure** — a calibrated win-probability model converts runs left, balls
   left, wickets remaining and the pre-match Elo gap into the chance the batting side wins.
   Pressure peaks when the contest is genuinely in the balance and is damped in dead games.

Each player's output in the top 30% highest-pressure deliveries — with a soft ramp around the
cutoff — is then measured against a **level-weighted expectation**: 85% the neutral situation
par and 15% the player's own usual level. The score is weighted by PI, shrunk for sample size
and adjusted for how often the player delivered a positive pressure innings. Putting most of
the weight on situation par means an elite player is judged mainly against the difficulty of
the situation rather than only against their own high bar, while the 15% own-baseline term
still rewards genuine improvement. The final lists are in Sections 5.1 and 5.2.

> **Headline result.** **Shreyas Iyer** and **Sunil Narine** lead the batting and bowling
> lists. The ranking is 85% a level test against situation par and 15% an improvement test
> against the player's own baseline, with a 400-ball floor and sample-size shrinkage.

**Headline selections**

{headline_table()}

The full top 10s, with pressure strike rates, economies, sample sizes and consistency, are in
Sections 5.1 and 5.2.

---

## 2. What the data contains

| File | Use |
|---|---|
| `ball_by_ball.csv` | Delivery-level outcome, batter, bowler, extras, dismissal |
| `matches.csv` | Competition, date, venue, teams, result, revised targets/overs |
| `competitions.csv` | Competition name, format and season |
| `teams.csv` | Team names |
| `playing_eleven.csv`, `competition_team_player_mapping.csv` | Player/team context (not needed for the core ratings) |

**Coverage by season.** The pack spans five calendar years and two formats. Every delivery
is used, including those outside the pressure set, because the own-baseline and situation-par
estimates need the full population.

{season_table()}

The pack contains **no ICC Men's T20 World Cup**. The senior multi-nation tournament present
is the **Asia Cup** (2022 and 2025); the development equivalents are the **Emerging Teams Asia
Cup 2024** and **Asia Cup Rising Stars 2025**. Because the pack has no World Cup, and because
the Asia Cup is not given World-Cup-level weight in this submission, all Asia Cup events sit
**below IPL league level** in the stakes ladder. The requested ordering is therefore preserved
where the data allows it: bilateral < IPL league < IPL playoffs, with the Asia Cup final above
bilateral cricket but below an IPL league match.

---

## 3. Defining pressure

### 3.1 Match stakes (S)

Every match is assigned a tier from its competition name and stage:

| Tier | Weight | Examples in the pack |
|---|---:|---|
| Bilateral T20I | {cfg['tier_weight']['bilateral']} | England in Pakistan T20I Series |
| Qualifier | {cfg['tier_weight']['qualifier']} | Asia Cup T20 Qualifiers |
| Emerging league | {cfg['tier_weight']['emerging_league']} | Emerging Teams Asia Cup group |
| Tri-series | {cfg['tier_weight']['tri_series']} | New Zealand T20I Tri Series |
| Asia Cup league | {cfg['tier_weight']['continental_league']} | Asia Cup group / Super 4 |
| Emerging knockout | {cfg['tier_weight']['emerging_knockout']} | Emerging Asia Cup semi/final |
| Asia Cup final | {cfg['tier_weight']['continental_knockout']} | Asia Cup final |
| IPL league | {cfg['tier_weight']['ipl_league']} | IPL 2022–2026 league stage |
| IPL playoff | {cfg['tier_weight']['ipl_playoff']} | Qualifier, Eliminator, Final |

This reproduces the requested ordering: bilateral < IPL league < IPL playoffs. The Asia Cup is
not elevated to the missing World Cup rung; its league stage is weighted alongside a tri-series
and its final sits above bilateral cricket but below IPL league level. IPL matches therefore
carry the highest stakes in the pack, and IPL playoffs carry the highest weight overall.

### 3.2 Opponent strength (O)

The pack has no ICC ranking table, so we compute a **pre-match Elo rating** for every team from
the results in the same data, separately for the IPL and international pools. The opponent's
pre-match Elo is converted to a 0–1 strength and mapped to a multiplier:

```
O = 0.80 + 0.40 * strength(opponent Elo)
```

so a player faces at most a ±20% stakes adjustment for opponent quality. Elo is updated after
each match (K = {cfg['elo_k']}, home advantage = {cfg['elo_home_adv']} Elo points). Using a
dynamic Elo rather than a single static ranking snapshot is deliberate: it tracks form and
opponent strength at the time of the match, and it is fully reproducible from the supplied data.

### 3.3 Situational pressure (P)

An earlier version of this model measured closeness with hand-set run-rate terms. That has a
known failure mode: in a chase that is already won (20 needed off 30 with eight wickets in
hand) the batting side is “ahead of par”, so a naive mirror-image formula calls it high bowling
pressure, while a genuine death-over squeeze (45 off 18 with four wickets) can score low. We
therefore replace the hand-set closeness terms with a **win-probability model fitted on the
supplied ball-by-ball data**.

For every delivery, before the ball is bowled, we form a small state vector:

- `adv` — the projected final score minus the target (2nd innings) or the venue par score
  (1st innings), divided by 20;
- `lost` — wickets already lost, divided by 10;
- `elapsed` — legal balls bowled divided by the innings length;
- `elo_diff` — the batting team's pre-match Elo minus the bowling team's, divided by 400;
- the interaction `adv * elapsed` and a constant.

A separate logistic model is fitted for each innings (`p = sigmoid(w · x)`), with the label
“did the batting team win?” and L2 regularisation. The fitted coefficients are:

| Innings | adv | lost | adv × elapsed | Elo diff | constant |
|---|---:|---:|---:|---:|---:|
| First | {wp1[0]:+.2f} | {wp1[1]:+.2f} | {wp1[2]:+.2f} | {wp1[3]:+.2f} | {wp1[4]:+.2f} |
| Second | {wp2[0]:+.2f} | {wp2[1]:+.2f} | {wp2[2]:+.2f} | {wp2[3]:+.2f} | {wp2[4]:+.2f} |

The model is used only to **order situations by leverage**, not as a claim that a player caused
a win-probability change. Leverage is the classic `4p(1-p)` term, which is 1 when the game is a
coin-flip and 0 when it is decided. We damp it early in the innings, when every game is still
nominally close but little has been decided:

```
p = sigmoid(w · x)
tension = 4 * p * (1 - p) * ({cfg['close_floor']} + (1 - {cfg['close_floor']}) * elapsed)
```

The two roles then get a different resources term, and — importantly — that term is **gated by
tension**:

```
wk_bat  = clip(wickets_lost / 9, 0, 1)      # a batter is under pressure when wickets fall
wk_bowl = clip(wickets_left / 10, 0, 1)     # a bowler is under pressure when wickets remain
P_bat   = tension * ({cfg['w_state']} + {cfg['w_wk']} * wk_bat)  + {cfg['w_ball']} * elapsed
P_bowl  = tension * ({cfg['w_state']} + {cfg['w_wk']} * wk_bowl) + {cfg['w_ball']} * elapsed
```

Gating the resources term is what removes the dead-game distortion completely. Wickets only
create pressure while the match is alive: once `p` has collapsed, `tension` goes to zero and
the resource term vanishes with it. The only term that survives a decided game is the small
late-innings term, which is why such deliveries fall below the pressure threshold instead of
inflating a bowler's tally.

Two worked chases (target 160, par 8.5) show the effect. A cakewalk — 20 needed off 30 with
eight wickets in hand — has `p ≈ 0.93`, `tension ≈ 0.22`, and scores **PI_bowl ≈ 34**. A
genuine squeeze — 45 needed off 18 with four wickets — has `p ≈ 0.10`, `tension ≈ 0.32`, and
scores **PI_bowl ≈ 41**. The tight chase now ranks higher for the bowler, which is the correct
ordering. A knife-edge game (`p ≈ 0.50`) has `tension ≈ 1.00` and dominates both. Runs left,
balls left and wickets remaining all still enter through the win-probability model and the
`elapsed` / resources terms, so the brief's situational inputs remain explicit.

### 3.4 Entry context

The situation a player walks into matters. For every innings we record the PI of the **first
ball faced by the batter** and the **first ball bowled by the bowler**. Each pressure delivery
weight is multiplied by

```
1 + {cfg['entry_bonus']} * entry_PI / 100
```

so a player who enters at 20/3 in the fifth over, or comes on in the death overs of a tight
chase, carries a modest additional weight through that innings. This directly implements the
“when the batter came in / when the bowler came on” requirement.

### 3.5 Pressure Index and threshold

The per-delivery index is

```
PI = 100 * S * O * P
```

PI is unbounded above; the observed range is about 0–137. A delivery is **core pressure** if
it is in the **top 30% of PI for that role** (70th percentile). In this dataset the thresholds
are:

| Role | 70th-percentile threshold | Pressure deliveries |
|---|---:|---:|
| Batting | {thr_bat:.1f} | {int(np.sum(pi_bat >= thr_bat)):,} of {len(pi_bat):,} balls faced |
| Bowling | {thr_bowl:.1f} | {int(np.sum(pi_bowl >= thr_bowl)):,} of {len(pi_bowl):,} legal balls |

![Pressure Index distribution](figures/fig_pressure_dist.png)

The percentile threshold defines the core pressure set, but the score does not use a hard
cutoff: Section 4.4 describes the soft logistic ramp that lets deliveries just below the
threshold contribute partial weight. Section 6 shows how sensitive the lists are to moving the
threshold.

---

## 4. Measuring performance

### 4.1 Baselines and expected performance

For each player we compute two expectations:

1. **Own baseline** — the player's overall runs per ball (batting) or runs conceded per legal
   ball (bowling) across the whole dataset. This is the “usual level” the brief asks for.
2. **Situation par** — the phase-specific par rate at the same point of the innings, adjusted
   for the venue. Phases are powerplay (first 30% of the innings), middle (30–75%) and death
   (last 25%). The fitted values are:

| Phase | Batting runs/ball | Batting wicket rate | Bowling runs/ball | Bowling wicket rate |
|---|---:|---:|---:|---:|
| Powerplay | {phase['bat']['0'][0]:.4f} | {phase['bat']['0'][1]:.4f} | {phase['bowl']['0'][0]:.4f} | {phase['bowl']['0'][1]:.4f} |
| Middle | {phase['bat']['1'][0]:.4f} | {phase['bat']['1'][1]:.4f} | {phase['bowl']['1'][0]:.4f} | {phase['bowl']['1'][1]:.4f} |
| Death | {phase['bat']['2'][0]:.4f} | {phase['bat']['2'][1]:.4f} | {phase['bowl']['2'][0]:.4f} | {phase['bowl']['2'][1]:.4f} |

The blended baseline used in the rating is

```
base = {cfg['absolute_weight']} * situation_par + (1 - {cfg['absolute_weight']}) * own_baseline
```

With `α = {cfg['absolute_weight']}`, this is **85% situation par and 15% own baseline**. For
batting, the per-ball impact can be written as

```
impact = {cfg['absolute_weight']} * (actual - par) + (1 - {cfg['absolute_weight']}) * (actual - own)
```

and for bowling the signs reverse:

```
impact = {cfg['absolute_weight']} * (par - conceded) + (1 - {cfg['absolute_weight']}) * (own - conceded)
```

So 85% of the judgement is the **level test** — did the player beat what a typical player would
be expected to do in that exact situation? — and 15% is the **improvement test** — did the player
beat their own usual level? Under the old 50/50 blend the own baseline carried half the weight,
which let a modest relative improver outrank an elite player who merely maintained an already
high level. At 85/15 the level test has clear precedence, while the own-baseline term stays
large enough that a player far above their own history still gains. Setting the weight to 1.0 would drop the own
baseline entirely and make the score pure performance against situation par; 0.85 deliberately
keeps a 15% improvement signal.

### 4.2 Batting impact

For every pressure ball faced:

```
impact_i = (runs_off_bat - base_runs_per_ball) - wv_i * (out - base_wicket_rate)
```

The wicket value is no longer a flat 5 runs. It varies with the phase, the dismissed batter's
position in the order, and how set that batter is:

```
wv_i = {cfg['wicket_value']} * order_mult * phase_mult * (1 + {cfg['wv_set_bonus']} * clip(batter_runs / 30, 0, 1))
```

`order_mult` runs from 1.60 for an opener to 0.20 for a No. 11; `phase_mult` is
{cfg['wv_phase'][0]} in the powerplay, {cfg['wv_phase'][1]} in the middle and {cfg['wv_phase'][2]}
at the death. Across the pack the average wicket value is about 7.5 runs, but the spread is
wide: a top-order powerplay wicket is worth roughly 10–13 runs, while a tailender dismissed in
the 20th over is worth roughly 1.5–3. This stops the model treating a wicket at 20/3 and a
wicket at 180/8 as the same event. The per-ball impacts are weighted by PI and the entry
factor.

### 4.3 Bowling impact

For every pressure legal ball:

```
impact_i = (base_runs_per_ball - runs_conceded) + wv_i * (wicket - base_wicket_rate)
```

A bowler is rewarded for conceding fewer runs than expected and for taking wickets above
expectation. The same per-ball wicket value applies to the bowling side: taking the wicket of a
set top-order batter is worth more than taking a tailender's. Byes and leg-byes are excluded
from runs conceded; wides and no-balls are included; wides and no-balls do not count as legal
balls.

### 4.4 Weighting, shrinkage and consistency

The score does not use a hard in/out cutoff. A delivery just below the 70th percentile is not
obviously less pressure than one just above it, so each delivery receives a soft weight:

```
soft_i = sigmoid((PI_i - threshold) / tau)
w_i = PI_i * soft_i * (1 + {cfg['entry_bonus']} * entry_PI / 100)
```

with `tau = {cfg['soft_tau_frac']} * threshold` (about {summary['soft_tau']['bat']:.1f} batting and
{summary['soft_tau']['bowl']:.1f} bowling PI points).
Deliveries down to `threshold - 2*tau` are included with a small weight; below that they are
dropped. The **core** pressure count (used for eligibility and the tables) still counts only
deliveries at or above the 70th percentile, so the soft band cannot be used to manufacture a
qualifying sample.

The raw pressure score is the weighted mean impact:

```
raw = sum(w_i * impact_i) / sum(w_i)
```

Small samples are noisy, so the score is shrunk towards zero:

```
ppi = raw * sum(w_i) / (sum(w_i) + {cfg['shrink_k']})
```

With K = {cfg['shrink_k']:.0f}, a player with 400 core pressure balls keeps about
400 / (400 + {cfg['shrink_k']:.0f}) ≈ {400/(400+cfg['shrink_k']):.2f} of the raw impact, while
one with 900 keeps {900/(900+cfg['shrink_k']):.2f}. Combined with the 400-ball eligibility
floor below, this prevents a short hot streak from topping a list. The full rankings also
report the standard error, `effective_balls` and `reliability` as diagnostics.

Finally, we measure **consistency**: the share of a player's pressure innings (minimum
{cfg['min_innings_balls']} pressure balls) in which the player's impact was positive. The final
score is

```
final = ppi * (1 - {cfg['consistency_weight']} + {cfg['consistency_weight']} * consistency)
```

so a player who repeatedly delivers in pressure innings is preferred to one with the same
average but a boom-or-bust record.

### 4.5 Eligibility

| Rule | Batters | Bowlers |
|---|---:|---:|
| Minimum total balls | {cfg['min_total_balls_bat']} | {cfg['min_total_balls_bowl']} |
| Minimum pressure balls | {cfg['min_pressure_balls_bat']} | {cfg['min_pressure_balls_bowl']} |
| Minimum pressure innings | {cfg['min_pressure_innings']} | {cfg['min_pressure_innings']} |

These thresholds leave **{summary['eligible_batters']} eligible batters** and
**{summary['eligible_bowlers']} eligible bowlers**. They deliberately exclude short cameos: a
player must have faced at least {cfg['min_pressure_balls_bat']} core pressure deliveries (or
bowled {cfg['min_pressure_balls_bowl']}) to enter a list. This floor removes thin but
eye-catching records. With the floor switched off, **Vaibhav Sooryavanshi** (93 pressure
balls) would top the batting list, followed by **Najibullah Zadran** (102) and
**Rilee Rossouw** (120); **Quinton de Kock** (268) would sit seventh. On the bowling side
**Dushan Hemantha** (57) and **Hasan Mahmud** (57) would lead. None of these is a fair
comparison with a 700-ball body of work, so the floor excludes them. Every player in the final
top 10 has at least
{min(int(r['pressure_balls']) for r in bat[:10] + bowl[:10])} pressure balls, so all clear the
floor comfortably.

Per the submission instruction, players representing **Pakistan** or **Pakistan A**
({summary['excluded_players']} players in the pack) are excluded from both rankings. Their
deliveries remain in the underlying data only as opposition context; no Pakistani player can
appear in either top 10.

---

## 5. Results

### 5.1 Top 10 batters under pressure

{bat_table(bat[:10])}

![Usual vs pressure strike rate for the top 10 batters](figures/fig_top_bat.png)

### 5.2 Top 10 bowlers under pressure

{bowl_table(bowl[:10])}

![Usual vs pressure economy for the top 10 bowlers](figures/fig_top_bowl.png)

### 5.3 Reading the lists

- **Shreyas Iyer** tops the batting list: {bat[0]['pressure_balls']} pressure balls across
  {bat[0]['pressure_innings']} pressure innings, a strike rate of {bat[0]['sr_pressure_actual']}
  against a usual {bat[0]['sr_baseline']} and a phase/venue par of {bat[0]['expected_sr']}. Both
  components are positive — {bat[0]['impact_runs']} runs per ball from scoring and
  {bat[0]['impact_wickets']} runs per ball from wickets — and the lead survives sample-size
  shrinkage and the consistency adjustment ({bat[0]['consistency']}).
- **Suryakumar Yadav** and **Shubman Gill** are effectively tied for second and third
  ({float(bat[1]['final_score']):.3f} vs {float(bat[2]['final_score']):.3f}). Surya has
  {bat[1]['pressure_balls']} pressure balls at a strike rate of {bat[1]['sr_pressure_actual']}
  against a usual {bat[1]['sr_baseline']} and par {bat[1]['expected_sr']}; Gill has
  {bat[2]['pressure_balls']} balls at {bat[2]['sr_pressure_actual']} against
  {bat[2]['sr_baseline']} and par {bat[2]['expected_sr']}. Surya's consistency
  ({bat[1]['consistency']}) is the highest in the batting top 10; Gill's is lower
  ({bat[2]['consistency']}) and his run impact is marginally smaller
  ({bat[2]['impact_runs']} vs {bat[1]['impact_runs']}).
- **Rajat Patidar** is fourth with the highest run impact in the list
  ({bat[3]['impact_runs']} per ball) and a pressure strike rate of
  {bat[3]['sr_pressure_actual']}, but on a smaller sample ({bat[3]['pressure_balls']} balls) and
  with almost no wicket impact ({bat[3]['impact_wickets']}).
- **Nicholas Pooran** is fifth ({bat[4]['pressure_balls']} balls, pressure strike rate
  {bat[4]['sr_pressure_actual']} against {bat[4]['sr_baseline']} usual and
  {bat[4]['expected_sr']} par), with the third-best wicket impact among batters
  ({bat[4]['impact_wickets']}). **Ruturaj Gaikwad** and **Jos Buttler** follow in sixth and
  seventh; Buttler's wicket impact ({bat[6]['impact_wickets']}) is the best among the batters.
- **Lokesh Rahul** is eighth ({bat[7]['pressure_balls']} balls, {bat[7]['sr_pressure_actual']}
  against {bat[7]['sr_baseline']} usual), **Heinrich Klaasen** ninth and **Ishan Kishan** tenth.
  Klaasen has the second-highest pressure strike rate ({bat[8]['sr_pressure_actual']}) and the
  largest batting sample ({bat[8]['pressure_balls']} balls), but his own baseline is already
  {bat[8]['sr_baseline']} and par is {bat[8]['expected_sr']}, so under the level-weighted bar
  his margin is smaller than the raw strike rate suggests. Ishan has a strong run impact
  ({bat[9]['impact_runs']}) but a negative wicket impact ({bat[9]['impact_wickets']}) and the
  lowest consistency in the top 10 ({bat[9]['consistency']}), which is why he sits tenth.
- **Sunil Narine** tops the bowling list with {bowl[0]['pressure_balls']} pressure balls, an
  economy of {bowl[0]['economy_pressure_actual']} against a usual
  {bowl[0]['economy_baseline']} and a par of {bowl[0]['expected_economy']}. His run impact is
  {bowl[0]['impact_runs']} per ball; the wicket component is slightly negative
  ({bowl[0]['impact_wickets']}), so his pressure wicket rate is a little below the blended bar,
  but his run suppression and consistency ({bowl[0]['consistency']}) carry him to first.
- **Jasprit Bumrah** is second ({bowl[bum_idx]['pressure_balls']} balls). His pressure economy
  of {bowl[bum_idx]['economy_pressure_actual']} is worse than his own usual
  {bowl[bum_idx]['economy_baseline']} but far better than the situation par of
  {bowl[bum_idx]['expected_economy']}, giving him the best run impact in the list
  ({bowl[bum_idx]['impact_runs']} per ball). The wicket component is
  {bowl[bum_idx]['impact_wickets']}: his pressure wicket rate is below the situation par, even
  though his raw wicket tally is high. Under the level-weighted blend his run suppression is
  enough for second; he is no longer judged mainly against his own elite baseline.
- **Matheesha Pathirana** and **Harshal Patel** are third and fourth
  ({float(bowl[2]['final_score']):.3f} vs {float(bowl[3]['final_score']):.3f}). Pathirana has the better
  absolute economy ({bowl[2]['economy_pressure_actual']} against
  {bowl[2]['economy_baseline']} usual and {bowl[2]['expected_economy']} par); Harshal has the
  larger sample ({bowl[3]['pressure_balls']} balls against {bowl[2]['pressure_balls']}) and a
  bigger improvement over his own level. Both have positive wicket impacts
  ({bowl[2]['impact_wickets']} and {bowl[3]['impact_wickets']}).
- **Harpreet Brar**, **Noor Ahmad** and **Wanindu Hasaranga** fill fifth to seventh. Hasaranga
  has the best wicket impact in the list ({bowl[6]['impact_wickets']}) but concedes more than his
  own baseline ({bowl[6]['economy_pressure_actual']} against
  {bowl[6]['economy_baseline']}), so his rank reflects the balance of the two components.
  **Krunal Pandya**, **Kuldeep Yadav** and **Varun Chakravarthy** complete the top 10; Kuldeep
  has the highest consistency in the bowling top 10 ({bowl[8]['consistency']}), while Varun's
  negative wicket impact ({bowl[9]['impact_wickets']}) leaves him tenth despite a large sample.
- The list is not simply a death-bowling award. Narine and Bumrah lead on run suppression across
  phases, while Pathirana and Harshal supply the death-over wicket threat. Brar, Noor and
  Krunal remain top-10 on large middle-overs samples, but the pressure model discounts tidy
  spells in games that are already decided.
- **What the 85/15 change did.** Moving from the old 50/50 blend to 85/15 raises Bumrah from
  fifth to second and drops Harshal from first to fourth; Narine inherits first. On the batting
  side Suryakumar Yadav rises from sixth to second, Klaasen enters at ninth and Pant drops out.
  This is the intended effect of giving the level test clear precedence over relative
  improvement.

## 6. Reliability and sensitivity

**Consistency check.** The `consistency` column is the fraction of pressure innings with a
positive impact. The top-10 batters range from
{min(float(r['consistency']) for r in bat[:10]):.2f} to {max(float(r['consistency']) for r in bat[:10]):.2f};
the top-10 bowlers range from
{min(float(r['consistency']) for r in bowl[:10]):.2f} to {max(float(r['consistency']) for r in bowl[:10]):.2f}.
No player reaches the top of a list on one extraordinary innings alone.

**Sensitivity.** Re-running the pipeline with different modelling choices keeps the leading
names but reshuffles the lower half of each top 10 (Jaccard overlap of the top-10 sets against
the base run):

| Variant | Batting overlap | Bowling overlap |
|---|---:|---:|
| Wicket value 2 instead of 5 | 1.00 | 1.00 |
| Wicket value 8 instead of 5 | 0.67 | 0.82 |
| Pressure = top 20% (not 30%) | 0.54 | 0.82 |
| Pressure = top 40% (not 30%) | 0.82 | 0.54 |
| Absolute weight 0.70 (not 0.85) | 0.82 | 0.82 |
| Absolute weight 1.00 (not 0.85) | 0.82 | 1.00 |
| Soft ramp tau 0.05 (not 0.10) | 1.00 | 1.00 |
| Soft ramp tau 0.20 (not 0.10) | 1.00 | 0.82 |

The lists are stable in composition at the top and more volatile in the lower half. Halving the
wicket value reproduces both top-10 sets exactly (the order changes); doubling it changes two
batting names and one bowling name, because it rewards wicket-taking over economy. Moving the level/improvement
blend to 0.70 or 1.00 keeps the same leading group and changes at most two names, which is the
expected behaviour: the blend changes the order and the back half, not the identity of the
best pressure players. The soft-ramp variants reproduce almost all of the base list. The
biggest movement comes from moving the pressure threshold to the top 20% or top 40%, which
changes what counts as pressure at all; even there the overlap is 0.54–0.82 rather than the
0.25 seen before the soft ramp. The lower half of each list should therefore be read as a
shortlist rather than a precise ordering.

> **Robustness.** The leading group is stable across every variant. The biggest movement
> comes from changing what counts as pressure, not from the level/improvement weight.

### Tier evidence

Pressure is not spread evenly across the calendar, and the tier weights show up in the data
before any player is scored. The mean PI rises from bilateral cricket to the IPL playoffs, and
the share of deliveries above the pressure threshold rises with it.

{tier_table()}

![Mean Pressure Index and match count by competition tier](figures/fig_tiers.png)

The table is the ladder made visible. IPL playoffs carry a mean batting PI of
{tier_mean('ipl_playoff', 'bat'):.1f} and place {tier_share('ipl_playoff'):.0%} of deliveries
above the pressure threshold, against {tier_mean('ipl_league', 'bat'):.1f} and
{tier_share('ipl_league'):.0%} in IPL league matches and {tier_mean('bilateral', 'bat'):.1f} and
{tier_share('bilateral'):.0%} in bilateral cricket. The Asia Cup final
({tier_mean('continental_knockout', 'bat'):.1f} batting,
{tier_mean('continental_knockout', 'bowl'):.1f} bowling) lands just below the IPL league,
exactly where the de-weighted ladder places it: above bilateral cricket, but not promoted to a
World Cup stand-in. The same ordering holds for bowling, where the playoff mean is
{tier_mean('ipl_playoff', 'bowl'):.1f} against {tier_mean('ipl_league', 'bowl'):.1f} in the
league stage.

Two consequences follow. The ladder is doing real work: the gaps are wide enough to survive
the win-probability and opponent adjustments. But because the IPL supplies
{tier_counts['ipl_league'] + tier_counts['ipl_playoff']} of the {summary['matches']} matches,
most of the high-pressure sample is IPL cricket. That is a property of the data pack, not a
thumb on the scale, but the lists are IPL-weighted evidence rather than a balanced global
sample.

---

## 7. Limitations

1. **No ICC World Cup in the pack.** The requested World Cup rung cannot be measured directly.
   The Asia Cup is present but is deliberately **not** treated as a World Cup substitute: it is
   weighted below IPL league level. If an ICC World Cup is added later, the tier mapping is a
   one-line change in `pipeline.py`.
2. **No official ICC ranking file.** Opponent strength uses a pre-match Elo rating computed
   from the same data. This is dynamic and reproducible, but it is a proxy, not the official
   ranking table.
3. **Pressure is a model, not a ground truth.** The 70th percentile defines the core pressure
   set and the soft ramp smooths the boundary; Section 6 shows the effect of moving both.
   Different weights produce somewhat different lists, which is why every parameter is
   documented in `outputs/config.json`. The score is measured against a level-weighted
   expectation (85% situation par, 15% the player's own baseline), so it is not a pure level
   ranking: 15% of the weight still rewards improvement on the player's own history, and a
   large relative jump can therefore move a player above a more established name. The
   `impact_runs` and `impact_wickets` columns make that trade-off visible for every player,
   and Section 6 shows how the lists respond when the weight is moved to 0.70 or 1.00.
4. **Sample sizes remain uneven.** A player with 400 core pressure balls has a wider
   uncertainty band than one with 900. The 400-ball floor and shrinkage reduce but do not
   eliminate this; the `pressure_balls`, `effective_balls` and `consistency` columns should be
   read alongside the rank.
5. **No player-level matchup quality.** Team strength is captured by pre-match Elo, but T20
   pressure is also matchup-driven: facing Bumrah and Rashid Khan in a tight chase is not the
   same as facing part-time bowlers in the same game state. The pack has no player-level Elo or
   matchup file, so this remains a documented limitation rather than a modelled input.
6. **The win-probability model is descriptive.** It is fitted on the same data and is used to
   order situations by leverage, not to claim that a player caused a particular
   win-probability change. It is a calibration device, not an externally validated forecasting
   model.
7. **Selection and role effects.** All-rounders can appear in both lists if they meet the
   volume thresholds. The lists are pressure-performance lists, not a claim about overall T20
   greatness.
8. **Rain and revised matches.** Revised targets and revised innings lengths are taken from
   `matches.csv` (the `*_revised_overs` fields store balls, not overs; this was verified
   against the delivery counts). Super overs are excluded.
9. **Pakistani players excluded.** Per the submission instruction, players representing
   Pakistan and Pakistan A are removed from the rankings. Under the current 400-ball floor no
   Pakistani player would have entered either top 10 anyway, so the exclusion does not change
   the submitted lists; it is applied as an explicit guard rather than an implicit one.
10. **Player labels.** Names are exactly as they appear in the data pack (for example
    “Namboori Tilak Varma”). The Unstop dropdowns may use shorter labels; map them carefully
    before submitting.

## 8. Reproducibility

```
python3 pipeline.py        # builds ratings and top-10 files
python3 sensitivity.py     # runs the sensitivity variants
python3 make_report.py     # builds charts and this report
```

| Output | Contents |
|---|---|
| `outputs/top10_batters.csv` | Final top 10 batters |
| `outputs/top10_bowlers.csv` | Final top 10 bowlers |
| `outputs/player_pressure_ratings_bat.csv` | Full eligible batting rankings |
| `outputs/player_pressure_ratings_bowl.csv` | Full eligible bowling rankings |
| `outputs/ball_pressure.csv` | Per-delivery PI and outcome (audit trail) |
| `outputs/summary.json` | Thresholds, phase par, tier counts and full config |

All parameters live in the `CFG` dictionary at the top of `pipeline.py`; `outputs/config.json`
records the exact values used for the submitted lists.
"""

md_path = os.path.join(OUT, "METHODOLOGY.md")
with open(md_path, "w") as fh:
    fh.write(report)

body_md = report.split("\n---\n", 1)[1]

sections = re.findall(r"^## (.+)$", body_md, flags=re.M)
section_descriptions = {
    "1. Executive summary": "The headline top 10s and the three inputs to the Pressure Index.",
    "2. What the data contains": "Coverage, competition mix, and the deliberate treatment of the Asia Cup.",
    "3. Defining pressure": "Stakes, opponent strength, win probability, and the gated leverage formula.",
    "4. Measuring performance": "Baselines, per-ball impact, sample shrinkage, consistency, and eligibility.",
    "5. Results": "The final batting and bowling lists, with the evidence behind each selection.",
    "6. Reliability and sensitivity": "Consistency checks, robustness to modelling choices, and tier evidence.",
    "7. Limitations": "What the model does not capture, and how to read the uncertainty.",
    "8. Reproducibility": "Commands, outputs, and the full configuration used for the submission.",
}


def decorate_html(html):
    """Add print-friendly classes to tables and turn images into numbered figures."""
    soup = BeautifulSoup(html, "html.parser")

    for idx, img in enumerate(soup.find_all("img"), 1):
        alt = (img.get("alt") or "").strip()
        parent = img.find_parent("p")
        figure = soup.new_tag("figure")
        if parent is not None:
            parent.replace_with(figure)
        else:
            img.insert_before(figure)
        figure.append(img)
        caption = soup.new_tag("figcaption")
        caption.string = f"Figure {idx}. {alt}" if alt else f"Figure {idx}."
        figure.append(caption)

    for table in soup.find_all("table"):
        classes = table.get("class", [])
        if "data-table" not in classes:
            classes.append("data-table")
        table["class"] = classes

        rows = table.find_all("tr")
        if not rows:
            continue
        if len(rows) <= 12:
            classes.append("keep")
            table["class"] = classes

        header_cells = rows[0].find_all(["th", "td"])
        if header_cells and header_cells[0].get_text(strip=True) == "Tier":
            classes.append("compact")
            table["class"] = classes
        header_cells = rows[0].find_all(["th", "td"])
        ncols = len(header_cells)

        rank_cols = set()
        for c, cell in enumerate(header_cells):
            if cell.get_text(strip=True).lower() == "rank":
                rank_cols.add(c)
                for row in rows:
                    cells = row.find_all(["td", "th"])
                    if c < len(cells):
                        cls = cells[c].get("class", [])
                        if "rank" not in cls:
                            cls.append("rank")
                        cells[c]["class"] = cls

        numeric_cols = []
        for c in range(ncols):
            if c in rank_cols:
                continue
            vals = []
            for row in rows[1:]:
                cells = row.find_all(["td", "th"])
                if c < len(cells):
                    txt = cells[c].get_text(" ", strip=True)
                    if txt:
                        vals.append(txt)
            if vals and all(
                re.fullmatch(r"[+\-−]?[\d,]+(?:\.\d+)?%?", v.replace(" ", ""))
                for v in vals
            ):
                numeric_cols.append(c)

        for c in numeric_cols:
            for row in rows:
                cells = row.find_all(["td", "th"])
                if c < len(cells):
                    cls = cells[c].get("class", [])
                    if "num" not in cls:
                        cls.append("num")
                    cells[c]["class"] = cls

    return str(soup)


def build_html(page_map=None):
    page_map = page_map or {}
    toc_rows = []
    for i, title in enumerate(sections, 1):
        desc = section_descriptions.get(title, "")
        if page_map:
            page = page_map.get(title, "—")
        else:
            page = "00"
        toc_rows.append(
            f'<tr><td class="toc-num">{i:02d}</td>'
            f'<td><div class="toc-title">{title}</div><div class="toc-desc">{desc}</div></td>'
            f'<td class="toc-page-num">{page}</td></tr>'
        )
    toc_rows = "\n".join(toc_rows)

    body_html = decorate_html(
        markdown.markdown(body_md, extensions=["tables", "fenced_code", "toc"])
    )

    cover = f"""
<section class="cover">
  <div class="cover-top">
    <div class="cover-brand">SupeRR <span>Selector</span></div>
    <div class="cover-round">Season 2 · Round 1</div>
  </div>
  <div class="cover-main">
    <div class="cover-eyebrow">Pressure-weighted T20 rankings</div>
    <h1 class="cover-title">Who holds up<br>when the game is tight?</h1>
    <p class="cover-subtitle">A reproducible ranking of the batters and bowlers who deliver when the match is on the line — built from {summary['matches']} matches and {summary['deliveries']:,} deliveries.</p>
    <div class="cover-rule"></div>
    <table class="cover-stats">
      <tr>
        <td><div class="stat-value">{summary['matches']}</div><div class="stat-label">matches</div></td>
        <td><div class="stat-value">{summary['deliveries']:,}</div><div class="stat-label">deliveries</div></td>
        <td><div class="stat-value">{summary['eligible_batters']} + {summary['eligible_bowlers']}</div><div class="stat-label">eligible players</div></td>
        <td><div class="stat-value">85 / 15</div><div class="stat-label">level / improvement</div></td>
      </tr>
    </table>
  </div>
  <div class="cover-bottom">
    <div class="cover-meta">
      <div><span>Prepared for</span> SupeRR Selector Season 2 · Round 1</div>
      <div><span>Data window</span> IPL 2022–2026 · T20 Internationals 2022–2026</div>
      <div><span>Date</span> November 2026</div>
    </div>
    <div class="cover-mark">RR</div>
  </div>
</section>
"""

    toc = f"""
<section class="toc-page">
  <div class="toc-kicker">Contents</div>
  <div class="toc-heading">What this report covers</div>
  <table class="toc-table">
    {toc_rows}
  </table>
  <table class="toc-highlights">
    <tr>
      <td><div class="toc-hl-label">Pressure Index</div><div class="toc-hl-value">Stakes × opponent × leverage</div><div class="toc-hl-desc">A calibrated win-probability model turns match state into a per-delivery pressure weight.</div></td>
      <td><div class="toc-hl-label">Level-weighted score</div><div class="toc-hl-value">85% par · 15% own</div><div class="toc-hl-desc">Most of the judgement is against the situation; a smaller term rewards improvement on the player's own level.</div></td>
      <td><div class="toc-hl-label">Sample guard</div><div class="toc-hl-value">400 balls · K = 600</div><div class="toc-hl-desc">Eligibility floor, shrinkage and a consistency adjustment stop short hot streaks from topping the lists.</div></td>
    </tr>
  </table>
  <div class="toc-note">All figures are generated directly from the supplied data pack. The per-delivery audit trail is in <code>outputs/ball_pressure.csv</code>.</div>
</section>
"""

    css = """
@page {
  size: A4;
  margin: 22mm 20mm 24mm 20mm;
  @top-left {
    content: "SUPERR SELECTOR · SEASON 2";
    font-family: "Montserrat", sans-serif;
    font-size: 6.8pt;
    letter-spacing: 0.12em;
    color: #64748b;
    border-bottom: 0.5pt solid #e2e8f0;
    padding-bottom: 3mm;
    vertical-align: bottom;
  }
  @top-right {
    content: "ROUND 1 · METHODOLOGY";
    font-family: "Montserrat", sans-serif;
    font-size: 6.8pt;
    letter-spacing: 0.12em;
    color: #64748b;
    border-bottom: 0.5pt solid #e2e8f0;
    padding-bottom: 3mm;
    vertical-align: bottom;
    text-align: right;
  }
  @bottom-left {
    content: "Pressure-weighted T20 rankings";
    font-family: "Carlito", sans-serif;
    font-size: 7.5pt;
    color: #94a3b8;
    border-top: 0.5pt solid #e2e8f0;
    padding-top: 3mm;
    vertical-align: top;
  }
  @bottom-right {
    content: "Page " counter(page) " of " counter(pages);
    font-family: "Montserrat", sans-serif;
    font-size: 7pt;
    letter-spacing: 0.04em;
    color: #64748b;
    border-top: 0.5pt solid #e2e8f0;
    padding-top: 3mm;
    vertical-align: top;
    text-align: right;
  }
}
@page :first {
  margin: 0;
  @top-left { content: ""; border: none; }
  @top-right { content: ""; border: none; }
  @bottom-left { content: ""; border: none; }
  @bottom-right { content: ""; border: none; }
}

* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body {
  margin: 0;
  font-family: "Carlito", "Noto Sans", sans-serif;
  font-size: 10.2pt;
  line-height: 1.55;
  color: #1e293b;
}
p { margin: 0 0 8pt; orphans: 3; widows: 3; }
strong { color: #0b1f3a; font-weight: 700; }
a { color: #0f766e; text-decoration: none; }

h1, h2, h3 {
  font-family: "Montserrat", "DejaVu Sans", sans-serif;
  color: #0b1f3a;
}
h2 {
  font-size: 17pt;
  font-weight: 700;
  line-height: 1.2;
  margin: 0 0 12pt;
  padding-bottom: 5pt;
  border-bottom: 1.2pt solid #dbe4ee;
  page-break-before: always;
  page-break-after: avoid;
}
h2::before {
  content: "";
  display: inline-block;
  width: 5pt;
  height: 16pt;
  background: #0f766e;
  margin-right: 8pt;
  vertical-align: -1pt;
}
h3 {
  font-size: 12pt;
  font-weight: 600;
  color: #0f766e;
  margin: 18pt 0 6pt;
  page-break-after: avoid;
}

table {
  width: 100%;
  border-collapse: collapse;
  margin: 10pt 0 14pt;
  font-size: 8.6pt;
  line-height: 1.35;
  page-break-inside: auto;
}
thead { display: table-header-group; }
tr { page-break-inside: avoid; }
th {
  background: #0b1f3a;
  color: #ffffff;
  font-family: "Montserrat", sans-serif;
  font-size: 7.8pt;
  font-weight: 600;
  letter-spacing: 0.03em;
  padding: 6pt 5pt;
  text-align: left;
  vertical-align: bottom;
}
td {
  padding: 5pt 5pt;
  border-bottom: 0.5pt solid #e2e8f0;
  vertical-align: top;
}
tbody tr:nth-child(even) td { background: #f8fafc; }
tbody tr:last-child td { border-bottom: 1pt solid #cbd5e1; }
th.num, td.num { text-align: right; font-variant-numeric: tabular-nums; }
th.rank, td.rank { text-align: center; }
table.data-table.keep { page-break-inside: avoid; }
p:has(+ table.keep) { page-break-after: avoid; }
table.data-table.compact td { padding: 3.2pt 5pt; }
table.data-table.compact th { padding: 5pt 5pt; }

code {
  font-family: "DejaVu Sans Mono", "Noto Sans Mono", monospace;
  font-size: 8.6pt;
  background: #eef2f7;
  color: #0b3b5a;
  padding: 1pt 4pt;
  border-radius: 3pt;
}
pre {
  background: #f8fafc;
  border: 0.5pt solid #e2e8f0;
  border-left: 3pt solid #0f766e;
  border-radius: 4pt;
  padding: 9pt 11pt;
  margin: 10pt 0 14pt;
  font-size: 8.4pt;
  line-height: 1.45;
  overflow-wrap: break-word;
  page-break-inside: avoid;
}
pre code { background: none; padding: 0; color: #1e293b; }

blockquote {
  margin: 12pt 0 14pt;
  padding: 10pt 14pt;
  background: #f0fdfa;
  border-left: 3.5pt solid #0f766e;
  border-radius: 4pt;
  color: #134e4a;
  page-break-inside: avoid;
}
blockquote p { margin: 0; }
blockquote strong { color: #0f766e; }

ul, ol { margin: 0 0 10pt; padding-left: 18pt; }
li { margin-bottom: 3pt; }

figure {
  margin: 12pt 0 16pt;
  page-break-inside: avoid;
  text-align: center;
}
figure img {
  max-width: 100%;
  height: auto;
  border: 0.5pt solid #e2e8f0;
  border-radius: 4pt;
  padding: 4pt;
  background: #ffffff;
}
figcaption {
  margin-top: 5pt;
  font-size: 8.4pt;
  color: #64748b;
  font-style: italic;
}

hr { display: none; }

/* ---------- Cover ---------- */
.cover {
  width: 210mm;
  height: 297mm;
  padding: 18mm 20mm 16mm;
  background:
    radial-gradient(circle at 86% 8%, rgba(20, 184, 166, 0.20), transparent 32%),
    radial-gradient(circle at 8% 94%, rgba(245, 158, 11, 0.12), transparent 30%),
    linear-gradient(155deg, #071528 0%, #0b1f3a 52%, #102a4c 100%);
  color: #ffffff;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  page-break-after: always;
  position: relative;
  overflow: hidden;
}
.cover::before {
  content: "";
  position: absolute;
  right: -38mm;
  top: 74mm;
  width: 118mm;
  height: 118mm;
  border: 0.6pt solid rgba(255, 255, 255, 0.08);
  border-radius: 50%;
}
.cover::after {
  content: "";
  position: absolute;
  right: -18mm;
  top: 94mm;
  width: 78mm;
  height: 78mm;
  border: 0.6pt solid rgba(20, 184, 166, 0.18);
  border-radius: 50%;
}
.cover > * { position: relative; z-index: 1; }
.cover-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.cover-brand {
  font-family: "Montserrat", sans-serif;
  font-size: 10.5pt;
  font-weight: 700;
  letter-spacing: 0.18em;
  text-transform: uppercase;
}
.cover-brand span { color: #5eead4; }
.cover-round {
  font-family: "Montserrat", sans-serif;
  font-size: 7.2pt;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #94a3b8;
}
.cover-main { margin-top: 26mm; max-width: 154mm; }
.cover-eyebrow {
  font-family: "Montserrat", sans-serif;
  font-size: 7.8pt;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: #5eead4;
  margin-bottom: 7mm;
}
.cover-title {
  font-family: "Montserrat", sans-serif;
  font-size: 33pt;
  font-weight: 800;
  line-height: 1.05;
  letter-spacing: -0.02em;
  color: #ffffff;
  margin: 0 0 7mm;
}
.cover-subtitle {
  font-family: "Carlito", "Noto Sans", sans-serif;
  font-size: 13pt;
  line-height: 1.5;
  color: #cbd5e1;
  max-width: 138mm;
  margin: 0 0 9mm;
}
.cover-rule {
  width: 22mm;
  height: 1.2mm;
  background: #f59e0b;
  margin-bottom: 9mm;
}
.cover-stats {
  width: 100%;
  border-collapse: collapse;
  margin: 0;
}
.cover-stats td {
  width: 25%;
  border-top: 0.6pt solid rgba(255, 255, 255, 0.22);
  border-bottom: none;
  padding: 5mm 6mm 0 0;
  vertical-align: top;
  background: none;
}
.stat-value {
  font-family: "Montserrat", sans-serif;
  font-size: 21pt;
  font-weight: 700;
  line-height: 1.1;
  color: #ffffff;
}
.stat-label {
  font-family: "Montserrat", sans-serif;
  font-size: 6.8pt;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #94a3b8;
  margin-top: 2mm;
}
.cover-bottom {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  border-top: 0.6pt solid rgba(255, 255, 255, 0.15);
  padding-top: 6mm;
}
.cover-meta {
  font-family: "Carlito", "Noto Sans", sans-serif;
  font-size: 8.5pt;
  line-height: 1.65;
  color: #cbd5e1;
}
.cover-meta span {
  display: inline-block;
  width: 27mm;
  font-family: "Montserrat", sans-serif;
  font-size: 6.6pt;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #94a3b8;
}
.cover-mark {
  width: 16mm;
  height: 16mm;
  border: 0.8pt solid rgba(255, 255, 255, 0.35);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: "Montserrat", sans-serif;
  font-size: 9pt;
  letter-spacing: 0.08em;
  color: #ffffff;
}

/* ---------- Contents ---------- */
.toc-page { padding-top: 0; }
.toc-kicker {
  font-family: "Montserrat", sans-serif;
  font-size: 7.8pt;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: #0f766e;
  margin-bottom: 4mm;
}
.toc-heading {
  font-family: "Montserrat", sans-serif;
  font-size: 24pt;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: #0b1f3a;
  margin: 0 0 6mm;
}
.toc-table {
  width: 100%;
  border-collapse: collapse;
  margin: 0 0 8mm;
}
.toc-table td {
  border-bottom: 0.5pt solid #e2e8f0;
  padding: 3mm 0;
  vertical-align: top;
  background: none;
}
.toc-table tbody tr:nth-child(even) td { background: none; }
.toc-num {
  width: 14mm;
  font-family: "Montserrat", sans-serif;
  font-size: 9pt;
  font-weight: 700;
  color: #0f766e;
  letter-spacing: 0.08em;
}
.toc-title {
  font-family: "Montserrat", sans-serif;
  font-size: 11.5pt;
  font-weight: 600;
  color: #0b1f3a;
  margin-bottom: 1.5mm;
}
.toc-desc {
  font-size: 9pt;
  line-height: 1.4;
  color: #64748b;
}
.toc-page-num {
  width: 10mm;
  text-align: right;
  font-family: "Montserrat", sans-serif;
  font-size: 9pt;
  color: #94a3b8;
}
.toc-highlights {
  width: 100%;
  border-collapse: separate;
  border-spacing: 3.5mm 0;
  margin: 0 0 6mm;
}
.toc-highlights td {
  width: 33.33%;
  background: #f8fafc;
  border-top: 2pt solid #0f766e;
  border-bottom: none;
  padding: 4mm;
  vertical-align: top;
}
.toc-hl-label {
  font-family: "Montserrat", sans-serif;
  font-size: 6.8pt;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #64748b;
  margin-bottom: 2mm;
}
.toc-hl-value {
  font-family: "Montserrat", sans-serif;
  font-size: 10.5pt;
  font-weight: 600;
  color: #0b1f3a;
  margin-bottom: 1.5mm;
}
.toc-hl-desc {
  font-size: 8.4pt;
  line-height: 1.4;
  color: #64748b;
}
.toc-note {
  font-size: 8.5pt;
  color: #64748b;
  border-top: 0.5pt solid #e2e8f0;
  padding-top: 3mm;
}
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SupeRR Selector Season 2 — Methodology</title>
<style>{css}</style>
</head>
<body>{cover}{toc}{body_html}</body>
</html>"""


html_path = os.path.join(OUT, "METHODOLOGY.html")
pdf_path = os.path.join(OUT, "METHODOLOGY.pdf")


def write_html(html):
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(html)


def render_pdf():
    """Render with Chrome for full CSS paged-media support; fall back to LibreOffice."""
    chrome = (
        shutil.which("google-chrome")
        or shutil.which("chromium")
        or shutil.which("chromium-browser")
    )
    if chrome:
        try:
            subprocess.run(
                [
                    chrome,
                    "--headless",
                    "--disable-gpu",
                    "--no-sandbox",
                    "--virtual-time-budget=3000",
                    "--no-pdf-header-footer",
                    f"--print-to-pdf={os.path.abspath(pdf_path)}",
                    f"file://{os.path.abspath(html_path)}",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return "chrome"
        except Exception:
            pass
    subprocess.run(
        [
            "soffice",
            "--headless",
            "-env:UserInstallation=file:///tmp/lo_profile",
            "--convert-to",
            "pdf",
            "--outdir",
            OUT,
            html_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return "libreoffice"


def detect_pages():
    """Find the printed page number of each section heading."""
    if not shutil.which("pdftotext"):
        return {}
    try:
        text = subprocess.run(
            ["pdftotext", "-layout", pdf_path, "-"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except Exception:
        return {}
    pages = text.split("\f")
    found = {}
    for title in sections:
        for i, page in enumerate(pages):
            if i < 2:
                continue
            if title in page:
                found[title] = i + 1
                break
    return found


write_html(build_html())
renderer = render_pdf()
page_map = detect_pages() if renderer == "chrome" else {}
if page_map:
    write_html(build_html(page_map))
    renderer = render_pdf()

print("wrote", md_path)
print("wrote", html_path)
print("wrote", pdf_path, f"({renderer})")
