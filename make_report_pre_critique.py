#!/usr/bin/env python3
"""Build charts, the methodology report (Markdown/HTML) and a PDF."""

import csv
import json
import os
import subprocess
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import markdown

import pipeline

OUT = "outputs"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
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
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    bins = np.linspace(0, 120, 49)
    ax.hist(pi_bat, bins=bins, alpha=0.55, label="Batting deliveries", color="#2b6cb0")
    ax.hist(pi_bowl, bins=bins, alpha=0.55, label="Bowling deliveries", color="#dd6b20")
    ax.axvline(thr_bat, color="#2b6cb0", ls="--", lw=1)
    ax.axvline(thr_bowl, color="#dd6b20", ls="--", lw=1)
    ax.text(thr_bat + 1, ax.get_ylim()[1] * 0.92, f"bat threshold {thr_bat:.1f}", color="#2b6cb0", fontsize=8)
    ax.text(thr_bowl + 1, ax.get_ylim()[1] * 0.80, f"bowl threshold {thr_bowl:.1f}", color="#dd6b20", fontsize=8)
    ax.set_xlabel("Pressure Index (PI)")
    ax.set_ylabel("Deliveries")
    ax.set_title("Distribution of the Pressure Index")
    ax.legend(frameon=False)
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
        c = "#2b6cb0"
        title = "Top 10 batters: usual vs pressure strike rate"
    else:
        base = [float(r["economy_baseline"]) for r in rows]
        press = [float(r["economy_pressure_actual"]) for r in rows]
        xlabel = "Economy rate (lower is better)"
        c = "#dd6b20"
        title = "Top 10 bowlers: usual vs pressure economy"
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    for yi, b, p in zip(y, base, press):
        ax.plot([b, p], [yi, yi], color="#bbbbbb", lw=2, zorder=1)
    ax.scatter(base, y, color="#888888", s=36, label="Usual", zorder=2)
    ax.scatter(press, y, color=c, s=46, label="Under pressure", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", color="#eeeeee")
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
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    axes[0].barh(labels, counts, color="#4a5568")
    axes[0].set_title("Matches by competition tier")
    axes[0].set_xlabel("Matches")
    for i, v in enumerate(counts):
        axes[0].text(v + 3, i, str(v), va="center", fontsize=8)
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].barh(labels, mean_bat, color="#2b6cb0")
    axes[1].set_title("Mean batting Pressure Index")
    axes[1].set_xlabel("Mean PI")
    for i, v in enumerate(mean_bat):
        axes[1].text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=8)
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
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
    lines = ["| Rank | Player | Team | Pressure balls | SR usual | SR pressure | SR par | PPI | Consistency |",
             "|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(
            f"| {r['rank']} | {r['player_name']} | {r['team']} | {r['pressure_balls']} | "
            f"{r['sr_baseline']} | {r['sr_pressure_actual']} | {r['expected_sr']} | "
            f"{float(r['final_score']):+.3f} | {r['consistency']} |")
    return "\n".join(lines)


def bowl_table(rows):
    lines = ["| Rank | Player | Team | Pressure balls | Econ usual | Econ pressure | Econ par | Wkts | PPI | Consistency |",
             "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(
            f"| {r['rank']} | {r['player_name']} | {r['team']} | {r['pressure_balls']} | "
            f"{r['economy_baseline']} | {r['economy_pressure_actual']} | {r['expected_economy']} | "
            f"{r['pressure_wickets']} | {float(r['final_score']):+.3f} | {r['consistency']} |")
    return "\n".join(lines)


cfg = summary["config"]
phase = summary["phase_par"]
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
3. **Situational pressure** — runs left / required run rate, wickets remaining, balls remaining,
   and how close the contest is.

Each player's output in the top 30% highest-pressure deliveries is then compared with a
**blended expectation** (50% situation par, 50% the player's own usual level), weighted by PI,
shrunk for sample size and adjusted for how often the player delivered a positive pressure
innings. The final lists are in Sections 5.1 and 5.2.

**Top 10 batters under pressure**

{bat_table(bat[:10])}

**Top 10 bowlers under pressure**

{bowl_table(bowl[:10])}

---

## 2. What the data contains

| File | Use |
|---|---|
| `ball_by_ball.csv` | Delivery-level outcome, batter, bowler, extras, dismissal |
| `matches.csv` | Competition, date, venue, teams, result, revised targets/overs |
| `competitions.csv` | Competition name, format and season |
| `teams.csv` | Team names |
| `playing_eleven.csv`, `competition_team_player_mapping.csv` | Player/team context (not needed for the core ratings) |

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

For each delivery, before the ball is bowled, we know:

- `runs_left` (2nd innings) or the deficit to the venue par score (1st innings),
- `balls_left`,
- `wkts_left`,
- the required run rate (`RRR`) or current run rate (`CRR`).

Venue par is the venue's first-innings scoring rate normalised to 120 balls (global fallback
when a venue has fewer than 1,200 first-innings balls). The par run rate is
`venue_par / 20`, clipped to 7.5–10.5.

**Batting pressure**

```
2nd innings:  rr = clip((RRR - par_RR) / {cfg['rr_scale']}, 0, 1)
1st innings:  rr = clip((par_score_at_stage - runs) / {cfg['deficit_scale']}, 0, 1)
wk = clip((10 - wkts_left) / 7, 0, 1)
bl = clip(balls_elapsed / total_balls, 0, 1)
close = clip(1 - |runs_left - par_RR * balls_left / 6| / {cfg['close_scale']}, 0, 1)
close_eff = close * ({cfg['close_floor']} + (1 - {cfg['close_floor']}) * bl)
P_bat = {cfg['w_rr']}*rr + {cfg['w_wk']}*wk + {cfg['w_ball']}*bl + {cfg['w_close']}*close_eff
```

**Bowling pressure** is the mirror image: `rr` rises when the batting side is *ahead* of par
(RRR below par in a chase, or CRR above par in the first innings) and `wk` rises when the
batting side still has wickets in hand. The same `bl` and `close_eff` terms apply. This means a
bowler is under pressure when defending a tight game, and a batter is under pressure when the
required rate is climbing or wickets are falling.

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

PI is unbounded above; the observed range is 0–135. A delivery is labelled **pressure** if it
is in the **top 30% of PI for that role** (70th percentile). In this dataset the thresholds are:

| Role | 70th-percentile threshold | Pressure deliveries |
|---|---:|---:|
| Batting | {thr_bat:.1f} | {int(np.sum(pi_bat >= thr_bat)):,} of {len(pi_bat):,} balls faced |
| Bowling | {thr_bowl:.1f} | {int(np.sum(pi_bowl >= thr_bowl)):,} of {len(pi_bowl):,} legal balls |

![Pressure Index distribution](figures/fig_pressure_dist.png)

The percentile threshold keeps the pressure sample large enough to rate players reliably while
still isolating the tense end of the distribution. Section 6 shows how sensitive the lists are
to moving this threshold.

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

The 50/50 blend stops the rating from being purely “most improved” (which would reward a
low-scoring player who improves slightly) or purely absolute (which would ignore the brief's
“compared with their usual performance levels”).

### 4.2 Batting impact

For every pressure ball faced:

```
impact = (runs_off_bat - base_runs_per_ball) - wicket_value * (out - base_wicket_rate)
```

`wicket_value = {cfg['wicket_value']}` runs. A batter is rewarded for scoring above expectation
and penalised for getting out more often than expected. The per-ball impacts are weighted by PI
and the entry factor.

### 4.3 Bowling impact

For every pressure legal ball:

```
impact = (base_runs_per_ball - runs_conceded) + wicket_value * (wicket - base_wicket_rate)
```

A bowler is rewarded for conceding fewer runs than expected and for taking wickets above
expectation. Byes and leg-byes are excluded from runs conceded; wides and no-balls are included;
wides and no-balls do not count as legal balls.

### 4.4 Weighting, shrinkage and consistency

The raw pressure score is the PI-weighted mean impact:

```
raw = sum(w_i * impact_i) / sum(w_i)
```

Small samples are noisy, so the score is shrunk towards zero:

```
ppi = raw * sum(w_i) / (sum(w_i) + {cfg['shrink_k']})
```

With K = {cfg['shrink_k']:.0f}, a player with 300 pressure balls keeps only
300 / (300 + {cfg['shrink_k']:.0f}) ≈ {300/(300+cfg['shrink_k']):.2f} of the raw impact, while
one with 900 keeps {900/(900+cfg['shrink_k']):.2f}. Combined with the 300-ball eligibility
floor below, this prevents a short hot streak from topping a list. The full rankings also
report the standard error and an `effective_balls` count as reliability diagnostics.

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
player must have faced at least {cfg['min_pressure_balls_bat']} pressure deliveries (or bowled
{cfg['min_pressure_balls_bowl']}) to enter a list. This floor removes thin but eye-catching
records — for example Jaker Ali Anik (201 pressure balls) and Anukul Roy (213) — which would
otherwise rank highly on a small sample. Every player in the final top 10 has at least
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

![Top 10 batters](figures/fig_top_bat.png)

### 5.2 Top 10 bowlers under pressure

{bowl_table(bowl[:10])}

![Top 10 bowlers](figures/fig_top_bowl.png)

### 5.3 Reading the lists

- **Shreyas Iyer** tops the batting list: {bat[0]['pressure_balls']} pressure balls across
  {bat[0]['pressure_innings']} pressure innings, a strike rate of {bat[0]['sr_pressure_actual']}
  against a usual {bat[0]['sr_baseline']} and a phase/venue par of {bat[0]['expected_sr']}. The
  improvement survives both the sample-size shrinkage and the consistency adjustment.
- **Jos Buttler** and **Suryakumar Yadav** combine high absolute pressure strike rates with
  large samples ({bat[bat_idx('Jos Buttler')]['pressure_balls']} and
  {bat[bat_idx('Suryakumar Yadav')]['pressure_balls']} pressure balls); they rank highly without
  needing a large “improvement” bonus.
- **Jasprit Bumrah** has the best absolute pressure economy in the bowling top 10
  ({bowl[bum_idx]['economy_pressure_actual']} vs a usual {bowl[bum_idx]['economy_baseline']});
  his ranking is held back only by a slightly lower pressure wicket rate than his own baseline.
- **Harpreet Brar**, **Jasprit Bumrah** and **Matheesha Pathirana** lead the bowling list on the
  strength of economy improvement plus wicket-taking in pressure overs; each has at least 400
  pressure balls.

---

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
| Wicket value 2 instead of 5 | 0.54 | 0.54 |
| Wicket value 8 instead of 5 | 0.82 | 0.67 |
| Pressure = top 20% (not 30%) | 0.43 | 0.25 |
| Pressure = top 40% (not 30%) | 0.43 | 0.43 |
| Absolute weight 0.30 (not 0.50) | 0.67 | 0.67 |
| Absolute weight 0.70 (not 0.50) | 0.67 | 0.67 |

The biggest movement comes from changing the pressure threshold, which is expected because that
changes which deliveries count as pressure at all. The lower half of each list should therefore
be read as a shortlist rather than a precise ordering.

**Tier evidence.** Pressure is not spread evenly across the calendar. The mean batting PI by
tier is shown below; IPL playoffs sit clearly above IPL league matches, and the Asia Cup final
sits above bilateral cricket but below IPL league level, as the de-weighted ladder intends.

![Tiers](figures/fig_tiers.png)

---

## 7. Limitations

1. **No ICC World Cup in the pack.** The requested World Cup rung cannot be measured directly.
   The Asia Cup is present but is deliberately **not** treated as a World Cup substitute: it is
   weighted below IPL league level. If an ICC World Cup is added later, the tier mapping is a
   one-line change in `pipeline.py`.
2. **No official ICC ranking file.** Opponent strength uses a pre-match Elo rating computed
   from the same data. This is dynamic and reproducible, but it is a proxy, not the official
   ranking table.
3. **Pressure is a model, not a ground truth.** The 30% threshold is a design choice; Section 6
   shows the effect of moving it. Different weights produce somewhat different lists, which is
   why the report documents every parameter in `outputs/config.json`.
4. **Sample sizes remain uneven.** A player with 300 pressure balls has a wider uncertainty
   band than one with 900. The 300-ball floor and shrinkage reduce but do not eliminate this;
   the `pressure_balls`, `effective_balls` and `consistency` columns should be read alongside
   the rank.
5. **Selection and role effects.** All-rounders can appear in both lists if they meet the
   volume thresholds. The lists are pressure-performance lists, not a claim about overall
   T20 greatness.
6. **Rain and revised matches.** Revised targets and revised innings lengths are taken from
   `matches.csv` (the `*_revised_overs` fields store balls, not overs; this was verified against
   the delivery counts). Super overs are excluded.
7. **Pakistani players excluded.** Per the submission instruction, players representing
   Pakistan and Pakistan A are removed from the rankings. This affects the bowling list in
   particular (Haris Rauf would otherwise have ranked fifth).
8. **Player labels.** Names are exactly as they appear in the data pack (for example
   “Namboori Tilak Varma”). The Unstop dropdowns may use shorter labels; map them carefully
   before submitting.

---

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

html_body = markdown.markdown(report, extensions=["tables", "fenced_code", "toc"])
css = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #1a202c; }
h1 { font-size: 22pt; color: #1a365d; border-bottom: 2px solid #2b6cb0; padding-bottom: 6px; }
h2 { font-size: 14pt; color: #1a365d; margin-top: 22px; }
h3 { font-size: 11.5pt; color: #2c5282; margin-top: 16px; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 14px; font-size: 8.6pt; }
th { background: #2b6cb0; color: white; padding: 5px; text-align: left; }
td { border-bottom: 1px solid #e2e8f0; padding: 4px 5px; }
tr:nth-child(even) td { background: #f7fafc; }
code, pre { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.8pt; background: #f1f5f9; }
pre { padding: 8px; border-left: 3px solid #2b6cb0; overflow-x: auto; }
img { max-width: 100%; }
blockquote { border-left: 3px solid #dd6b20; margin-left: 0; padding-left: 12px; color: #4a5568; }
"""
html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{css}</style></head>
<body>{html_body}</body></html>"""
html_path = os.path.join(OUT, "METHODOLOGY.html")
with open(html_path, "w") as fh:
    fh.write(html)

# PDF via LibreOffice
try:
    subprocess.run([
        "soffice", "--headless",
        "-env:UserInstallation=file:///tmp/lo_profile",
        "--convert-to", "pdf", "--outdir", OUT, html_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("wrote", os.path.join(OUT, "METHODOLOGY.pdf"))
except Exception as exc:
    print("PDF conversion failed:", exc)

print("wrote", md_path)
print("wrote", html_path)
