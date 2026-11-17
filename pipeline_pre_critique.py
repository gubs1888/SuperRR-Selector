#!/usr/bin/env python3
"""
SupeRR Selector Season 2 - Round 1
Pressure-weighted batting / bowling ratings from ball-by-ball data.

Pipeline
--------
1. Classify every match into a stakes tier (bilateral < league < knockout).
2. Build a pre-match Elo rating per team (separate pools for IPL and international)
   and use the opponent's pre-match Elo as a proxy for ICC ranking strength.
3. Walk every delivery and compute a situational pressure index from
   runs left / required rate, wickets left, balls left and closeness.
4. Keep the top 30% highest-pressure deliveries for each role, weight each
   player's performance by that pressure and compare it with a blended
   expectation (50% situation par, 50% the player's own baseline).
5. Shrink small samples, reward consistency across pressure innings and rank.

Only the Python standard library is used.
"""

import csv
import json
import math
import os
import statistics
from collections import Counter, defaultdict

DATA_DIR = "Dataset/Data Pack"
OUT_DIR = "outputs"

# ----------------------------------------------------------------------------
# Configuration (all tunable in one place)
# ----------------------------------------------------------------------------
CFG = {
    # stakes ladder: bilateral < league (= IPL league / continental league)
    #                < knockout (= IPL playoffs / continental final)
    "tier_weight": {
        "bilateral": 0.60,
        "qualifier": 0.60,
        "tri_series": 0.70,
        "emerging_league": 0.60,
        "emerging_knockout": 0.75,
        "ipl_league": 1.00,
        "ipl_playoff": 1.40,
        "continental_league": 0.70,
        "continental_knockout": 0.85,
    },
    # opponent strength multiplier = lo + (hi - lo) * strength(0..1)
    "opp_mult_lo": 0.80,
    "opp_mult_hi": 1.20,
    "elo_k": 24.0,
    "elo_home_adv": 20.0,
    "elo_start": 1500.0,

    # situational pressure formula (weights sum to 1)
    "par_rr_default": 8.5,
    "rr_scale": 6.0,          # runs-per-over away from par that maps to 1
    "deficit_scale": 30.0,    # runs behind par that maps to 1
    "close_scale": 40.0,      # runs away from par that maps to 0 closeness
    "w_rr": 0.35,
    "w_wk": 0.25,
    "w_ball": 0.15,
    "w_close": 0.25,
    "close_floor": 0.40,      # closeness weight early in the innings
    "entry_bonus": 0.15,      # weight boost from the pressure when a player entered

    # player ratings
    "wicket_value": 5.0,     # runs-equivalent value of one wicket
    "absolute_weight": 0.50,  # weight on situation-par performance vs own baseline
    "pressure_percentile": 70,  # top 30% of deliveries per role = pressure
    "shrink_k": 600.0,        # empirical-Bayes shrinkage (weighted balls)
    "consistency_weight": 0.20,
    "min_total_balls_bat": 300,
    "min_pressure_balls_bat": 300,
    "min_total_balls_bowl": 240,
    "min_pressure_balls_bowl": 300,
    "min_pressure_innings": 5,
    "min_innings_balls": 6,

    # submission instruction: ignore players representing these teams
    "exclude_teams": ["Pakistan", "Pakistan A"],
}

BOWLER_WKT = {"Caught", "Bowled", "LBW", "Stump", "Hit Wicket"}
NO_BALL = {"Overstep", "other"}
NOT_A_BALL = {"wide", "Penalty"} | NO_BALL


def clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def read_csv(name):
    with open(os.path.join(DATA_DIR, name), newline="") as fh:
        return list(csv.DictReader(fh))


def percentile(values, pct):
    if not values:
        return 0.0
    vals = sorted(values)
    k = (len(vals) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(vals) - 1)
    frac = k - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def classify_match(comp_name, stage):
    """Return (tier_key, elo_pool) for a match."""
    name = comp_name.lower()
    st = (stage or "").lower()
    knockout = st in {"final", "semi final", "qualifier", "eliminator"}

    if "ipl" in name:
        return ("ipl_playoff" if knockout else "ipl_league"), "ipl"
    if "qualifier" in name:
        return "qualifier", "international"
    if "asia cup" in name and "emerging" not in name and "rising" not in name:
        return ("continental_knockout" if knockout else "continental_league"), "international"
    if "emerging" in name or "rising stars" in name:
        return ("emerging_knockout" if knockout else "emerging_league"), "international"
    if "tri-series" in name or "tri series" in name or "tri-nation" in name:
        return "tri_series", "international"
    return "bilateral", "international"


def strength_from_elo(elo):
    """Map an Elo rating to 0..1 (0.5 at 1500)."""
    return 1.0 / (1.0 + 10.0 ** (-(elo - CFG["elo_start"]) / 400.0))


def build_match_meta():
    comps = {r["comp_id"]: r for r in read_csv("competitions.csv")}
    teams = {r["team_id"]: r["team_name"] for r in read_csv("teams.csv")}

    matches = read_csv("matches.csv")
    matches.sort(key=lambda r: (r["match_date"], r["match_id"]))

    meta = {}
    for m in matches:
        comp = comps.get(m["comp_id"], {"comp_name": "", "match_format": "", "season": ""})
        tier, pool = classify_match(comp["comp_name"], m["stage"])
        meta[m["match_id"]] = {
            "comp_name": comp["comp_name"],
            "format": comp["match_format"],
            "season": comp["season"],
            "date": m["match_date"],
            "stage": m["stage"],
            "venue_id": m["venue_id"],
            "venue_country": m["venue_country"],
            "home": m["home_team_id"],
            "away": m["away_team_id"],
            "home_name": teams.get(m["home_team_id"], m["home_team"]),
            "away_name": teams.get(m["away_team_id"], m["away_team"]),
            "winner": m["win_team_id"] or None,
            "result": m["result"],
            "first_rev": int(m["first_inn_revised_overs"]) if m["first_inn_revised_overs"] else None,
            "second_rev": int(m["second_inn_revised_overs"]) if m["second_inn_revised_overs"] else None,
            "target": int(m["revised_target"]) if m["revised_target"] else None,
            "tier": tier,
            "pool": pool,
            "tier_weight": CFG["tier_weight"][tier],
        }

    # ---- Elo, computed chronologically inside each pool --------------------
    elo = defaultdict(lambda: CFG["elo_start"])
    for mid, m in meta.items():
        pool = m["pool"]
        rh = elo[(pool, m["home"])]
        ra = elo[(pool, m["away"])]
        m["elo_home_pre"] = rh
        m["elo_away_pre"] = ra
        m["opp_strength_home"] = strength_from_elo(ra)
        m["opp_strength_away"] = strength_from_elo(rh)

        if m["winner"]:
            exp_home = 1.0 / (1.0 + 10.0 ** (-(rh + CFG["elo_home_adv"] - ra) / 400.0))
            score_home = 1.0 if m["winner"] == m["home"] else 0.0
            delta = CFG["elo_k"] * (score_home - exp_home)
            elo[(pool, m["home"])] = rh + delta
            elo[(pool, m["away"])] = ra - delta

    return meta


def build_innings(match_meta, ball_rows):
    """Innings totals, targets and balls available."""
    runs = defaultdict(int)
    legal = defaultdict(int)
    for r in ball_rows:
        if r["innings"] not in ("1", "2"):
            continue
        key = (r["match_id"], r["innings"])
        runs[key] += int(r["runs"]) + int(r["extras1_runs"] or 0) + int(r["extras_2_runs"] or 0)
        if r["extras_type"] not in NOT_A_BALL:
            legal[key] += 1

    innings = {}
    for (mid, inn), total in runs.items():
        m = match_meta[mid]
        if inn == "1":
            total_balls = m["first_rev"] or 120
            target = None
        else:
            total_balls = m["second_rev"] or 120
            target = m["target"] or (runs.get((mid, "1"), 0) + 1)
        innings[(mid, inn)] = {
            "runs": total,
            "legal_balls": legal[(mid, inn)],
            "total_balls": total_balls,
            "target": target,
        }
    return innings


def venue_par_scores(innings, match_meta):
    """Venue par first-innings score per 120 balls, and global fallback."""
    agg = defaultdict(lambda: [0, 0])
    global_agg = [0, 0]
    for (mid, inn), d in innings.items():
        if inn != "1":
            continue
        vid = match_meta[mid]["venue_id"]
        agg[vid][0] += d["runs"]
        agg[vid][1] += d["legal_balls"]
        global_agg[0] += d["runs"]
        global_agg[1] += d["legal_balls"]

    global_par = 120.0 * global_agg[0] / max(global_agg[1], 1)
    par = {}
    for vid, (r, b) in agg.items():
        par[vid] = 120.0 * r / b if b >= 1200 else global_par
    return par, global_par


def compute_deliveries(match_meta, ball_rows, innings, venue_par, global_par):
    """Walk every delivery once and record pressure indices, outcomes and phase par."""
    by_innings = defaultdict(list)
    for r in ball_rows:
        by_innings[(r["match_id"], r["innings"])].append(r)
    for k in by_innings:
        by_innings[k].sort(key=lambda r: int(r["innings_ball_num"]))

    records = []
    totals = {
        "bat": defaultdict(lambda: {"balls": 0, "runs": 0, "outs": 0, "innings": set(),
                                    "matches": set(), "team": Counter()}),
        "bowl": defaultdict(lambda: {"balls": 0, "runs": 0, "wkts": 0, "innings": set(),
                                     "matches": set(), "team": Counter()}),
    }
    pi_values = {"bat": [], "bowl": []}
    phase_stats = {
        "bat": defaultdict(lambda: [0, 0, 0]),   # balls, runs, outs
        "bowl": defaultdict(lambda: [0, 0, 0]),  # balls, runs, wickets
    }

    for (mid, inn), rows in by_innings.items():
        m = match_meta[mid]
        d_inn = innings[(mid, inn)]
        total_balls = d_inn["total_balls"]
        target = d_inn["target"]
        par_score = venue_par.get(m["venue_id"], global_par)
        par_rr = clip(par_score / 20.0, 7.5, 10.5)
        venue_factor = clip(par_score / global_par, 0.85, 1.15)

        opp_mult_home = CFG["opp_mult_lo"] + (CFG["opp_mult_hi"] - CFG["opp_mult_lo"]) * m["opp_strength_home"]
        opp_mult_away = CFG["opp_mult_lo"] + (CFG["opp_mult_hi"] - CFG["opp_mult_lo"]) * m["opp_strength_away"]

        runs_before = 0
        wkts_before = 0
        legal_before = 0

        for r in rows:
            et = r["extras_type"]
            runs_off_bat = int(r["runs"])
            e1 = int(r["extras1_runs"] or 0)
            e2 = int(r["extras_2_runs"] or 0)
            conceded = runs_off_bat + (e1 + e2 if et in ({"wide"} | NO_BALL) else 0)
            is_legal = et not in NOT_A_BALL
            faced = et not in ("wide", "Penalty")
            dismissal = (r["dismissal_type"] or "").strip()
            batter_out = bool(dismissal) and r["dismissed_player_id"] == r["striker_id"]
            bowler_wkt = dismissal in BOWLER_WKT

            balls_left = max(total_balls - legal_before, 0)
            wkts_left = max(10 - wkts_before, 0)
            bl = clip((total_balls - balls_left) / max(total_balls, 1))

            if inn == "1":
                crr = runs_before / max(legal_before / 6.0, 0.1)
                par_stage = par_score * (legal_before / max(total_balls, 1))
                deficit = par_stage - runs_before
                rr_bat = clip(deficit / CFG["deficit_scale"])
                rr_bowl = clip((crr - par_rr) / CFG["rr_scale"])
                close = clip(1.0 - abs(deficit) / CFG["close_scale"])
            else:
                runs_left = max(target - runs_before, 0)
                rrr = runs_left / max(balls_left / 6.0, 0.1)
                rr_bat = clip((rrr - par_rr) / CFG["rr_scale"])
                rr_bowl = clip((par_rr - rrr) / CFG["rr_scale"])
                close = clip(1.0 - abs(runs_left - par_rr * balls_left / 6.0) / CFG["close_scale"])

            wk_bat = clip((10 - wkts_left) / 7.0)
            wk_bowl = clip((wkts_left - 2) / 8.0)
            close_eff = close * (CFG["close_floor"] + (1.0 - CFG["close_floor"]) * bl)

            p_bat = (CFG["w_rr"] * rr_bat + CFG["w_wk"] * wk_bat
                     + CFG["w_ball"] * bl + CFG["w_close"] * close_eff)
            p_bowl = (CFG["w_rr"] * rr_bowl + CFG["w_wk"] * wk_bowl
                      + CFG["w_ball"] * bl + CFG["w_close"] * close_eff)

            opp_mult = opp_mult_home if r["batting_team_id"] == m["home"] else opp_mult_away
            stakes = m["tier_weight"] * opp_mult
            pi_bat = 100.0 * stakes * p_bat
            pi_bowl = 100.0 * stakes * p_bowl

            phase = 0 if bl < 0.30 else (1 if bl < 0.75 else 2)

            sid, bid = r["striker_id"], r["bowler_id"]

            bt = totals["bat"][sid]
            bt["balls"] += 1 if faced else 0
            bt["runs"] += runs_off_bat
            bt["outs"] += 1 if batter_out else 0
            bt["innings"].add((mid, inn))
            bt["matches"].add(mid)
            bt["team"][r["batting_team_id"]] += 1

            bo = totals["bowl"][bid]
            bo["balls"] += 1 if is_legal else 0
            bo["runs"] += conceded
            bo["wkts"] += 1 if bowler_wkt else 0
            bo["innings"].add((mid, inn))
            bo["matches"].add(mid)
            bo["team"][r["bowling_team_id"]] += 1

            if faced:
                ps = phase_stats["bat"][phase]
                ps[0] += 1
                ps[1] += runs_off_bat
                ps[2] += 1 if batter_out else 0
                pi_values["bat"].append(pi_bat)
            if is_legal:
                ps = phase_stats["bowl"][phase]
                ps[0] += 1
                ps[1] += conceded
                ps[2] += 1 if bowler_wkt else 0
                pi_values["bowl"].append(pi_bowl)

            records.append((mid, inn, sid, bid, runs_off_bat, conceded, faced, is_legal,
                            batter_out, bowler_wkt, pi_bat, pi_bowl, phase, venue_factor))

            runs_before += runs_off_bat + e1 + e2
            if is_legal:
                legal_before += 1
            if dismissal:
                wkts_before += 1

    phase_par = {
        "bat": {p: (s[1] / s[0], s[2] / s[0]) for p, s in phase_stats["bat"].items() if s[0]},
        "bowl": {p: (s[1] / s[0], s[2] / s[0]) for p, s in phase_stats["bowl"].items() if s[0]},
    }
    return records, totals, pi_values, phase_par


def aggregate_pressure(records, thresholds, phase_par):
    """Second pass: pressure-weighted player aggregates."""
    # cell = [weight, weighted_runs, weighted_wkts/outs, count, weighted_expected_runs, weighted_expected_wkts]
    bat = defaultdict(lambda: {"w": 0.0, "wruns": 0.0, "wouts": 0.0, "n": 0, "runs": 0, "outs": 0,
                               "wexp_runs": 0.0, "wexp_wkt": 0.0, "samples": [],
                               "inn": defaultdict(lambda: [0.0, 0.0, 0.0, 0, 0.0, 0.0])})
    bowl = defaultdict(lambda: {"w": 0.0, "wruns": 0.0, "wwkts": 0.0, "n": 0, "runs": 0, "wkts": 0,
                                "wexp_runs": 0.0, "wexp_wkt": 0.0, "samples": [],
                                "inn": defaultdict(lambda: [0.0, 0.0, 0.0, 0, 0.0, 0.0])})
    entry_bat = {}
    entry_bowl = {}

    for (mid, inn, sid, bid, runs_off_bat, conceded, faced, is_legal,
         batter_out, bowler_wkt, pi_bat, pi_bowl, phase, venue_factor) in records:
        if faced:
            entry_bat.setdefault((sid, mid, inn), pi_bat)
        if is_legal:
            entry_bowl.setdefault((bid, mid, inn), pi_bowl)

        if faced and pi_bat >= thresholds["bat"]:
            par_rpb, par_wkt = phase_par["bat"][phase]
            exp_runs = par_rpb * venue_factor
            w = pi_bat * (1.0 + CFG["entry_bonus"] * entry_bat[(sid, mid, inn)] / 100.0)
            a = bat[sid]
            a["w"] += w
            a["wruns"] += w * runs_off_bat
            a["wouts"] += w * (1 if batter_out else 0)
            a["wexp_runs"] += w * exp_runs
            a["wexp_wkt"] += w * par_wkt
            a["n"] += 1
            a["runs"] += runs_off_bat
            a["outs"] += 1 if batter_out else 0
            a["samples"].append((w, runs_off_bat, 1 if batter_out else 0, exp_runs, par_wkt))
            cell = a["inn"][(mid, inn)]
            cell[0] += w
            cell[1] += w * runs_off_bat
            cell[2] += w * (1 if batter_out else 0)
            cell[3] += 1
            cell[4] += w * exp_runs
            cell[5] += w * par_wkt

        if is_legal and pi_bowl >= thresholds["bowl"]:
            par_rpb, par_wkt = phase_par["bowl"][phase]
            exp_runs = par_rpb * venue_factor
            w = pi_bowl * (1.0 + CFG["entry_bonus"] * entry_bowl[(bid, mid, inn)] / 100.0)
            a = bowl[bid]
            a["w"] += w
            a["wruns"] += w * conceded
            a["wwkts"] += w * (1 if bowler_wkt else 0)
            a["wexp_runs"] += w * exp_runs
            a["wexp_wkt"] += w * par_wkt
            a["n"] += 1
            a["runs"] += conceded
            a["wkts"] += 1 if bowler_wkt else 0
            a["samples"].append((w, conceded, 1 if bowler_wkt else 0, exp_runs, par_wkt))
            cell = a["inn"][(mid, inn)]
            cell[0] += w
            cell[1] += w * conceded
            cell[2] += w * (1 if bowler_wkt else 0)
            cell[3] += 1
            cell[4] += w * exp_runs
            cell[5] += w * par_wkt

    return bat, bowl


def blended_impact(kind, pressure_rpb, pressure_wkt, own_rpb, own_wkt,
                   exp_runs_rpb, exp_wkt, wv):
    """50% situation-par baseline, 50% player's own baseline."""
    base_rpb = CFG["absolute_weight"] * exp_runs_rpb + (1 - CFG["absolute_weight"]) * own_rpb
    base_wkt = CFG["absolute_weight"] * exp_wkt + (1 - CFG["absolute_weight"]) * own_wkt
    if kind == "bat":
        return (pressure_rpb - base_rpb) - wv * (pressure_wkt - base_wkt)
    return (base_rpb - pressure_rpb) + wv * (pressure_wkt - base_wkt)


def weighted_impact_stats(samples, base_rpb, base_wkt, wv, kind):
    """Weighted mean impact, its standard error and effective sample size."""
    pairs = []
    sw = sw2 = swx = 0.0
    for w, runs, wkt, _exp_runs, _exp_wkt in samples:
        if kind == "bat":
            x = (runs - base_rpb) - wv * (wkt - base_wkt)
        else:
            x = (base_rpb - runs) + wv * (wkt - base_wkt)
        pairs.append((w, x))
        sw += w
        sw2 += w * w
        swx += w * x
    mean = swx / sw
    var = sum(w * (x - mean) ** 2 for w, x in pairs) / sw
    n_eff = sw * sw / sw2 if sw2 else 0.0
    se = math.sqrt(var / n_eff) if n_eff > 0 else 0.0
    return mean, se, n_eff


def run_pipeline():
    os.makedirs(OUT_DIR, exist_ok=True)

    match_meta = build_match_meta()
    ball_rows = [r for r in read_csv("ball_by_ball.csv") if r["innings"] in ("1", "2")]
    innings = build_innings(match_meta, ball_rows)
    venue_par, global_par = venue_par_scores(innings, match_meta)

    records, totals, pi_values, phase_par = compute_deliveries(
        match_meta, ball_rows, innings, venue_par, global_par)

    thresholds = {role: percentile(pi_values[role], CFG["pressure_percentile"])
                  for role in ("bat", "bowl")}
    bat_press, bowl_press = aggregate_pressure(records, thresholds, phase_par)

    names = {}
    for r in ball_rows:
        names.setdefault(r["striker_id"], r["striker_name"])
        names.setdefault(r["bowler_id"], r["bowler_name"])
    teams = {r["team_id"]: r["team_name"] for r in read_csv("teams.csv")}
    excluded_ids = {tid for tid, name in teams.items() if name in CFG["exclude_teams"]}

    def team_name(counter):
        if not counter:
            return ""
        tid = counter.most_common(1)[0][0]
        return teams.get(tid, tid)

    def consistency(inn_map, own_rpb, own_wkt, kind):
        vals = []
        for cell in inn_map.values():
            w, wruns, wkts, n, wexp_runs, wexp_wkt = cell
            if n >= CFG["min_innings_balls"] and w > 0:
                impact = blended_impact(kind, wruns / w, wkts / w, own_rpb, own_wkt,
                                        wexp_runs / w, wexp_wkt / w, CFG["wicket_value"])
                vals.append(impact)
        if not vals:
            return None, 0, 0.0
        frac = sum(1 for v in vals if v > 0) / len(vals)
        return frac, len(vals), statistics.pstdev(vals) if len(vals) > 1 else 0.0

    bat_rows = []
    excluded_players = set()
    for pid, t in totals["bat"].items():
        if any(tid in excluded_ids for tid in t["team"]):
            excluded_players.add(pid)
            continue
        if t["balls"] < CFG["min_total_balls_bat"]:
            continue
        p = bat_press.get(pid)
        if not p or p["n"] < CFG["min_pressure_balls_bat"]:
            continue
        own_rpb = t["runs"] / t["balls"]
        own_wkt = t["outs"] / t["balls"]
        pressure_rpb = p["wruns"] / p["w"]
        pressure_wkt = p["wouts"] / p["w"]
        exp_rpb = p["wexp_runs"] / p["w"]
        exp_wkt = p["wexp_wkt"] / p["w"]
        base_rpb = CFG["absolute_weight"] * exp_rpb + (1 - CFG["absolute_weight"]) * own_rpb
        base_wkt = CFG["absolute_weight"] * exp_wkt + (1 - CFG["absolute_weight"]) * own_wkt
        impact, se, n_eff = weighted_impact_stats(
            p["samples"], base_rpb, base_wkt, CFG["wicket_value"], "bat")
        ppi = impact * p["w"] / (p["w"] + CFG["shrink_k"])
        cons, n_inn, sd = consistency(p["inn"], own_rpb, own_wkt, "bat")
        if n_inn < CFG["min_pressure_innings"]:
            continue
        final = ppi * (1.0 - CFG["consistency_weight"] + CFG["consistency_weight"] * cons)
        bat_rows.append({
            "player_id": pid,
            "player_name": names.get(pid, pid),
            "team": team_name(t["team"]),
            "matches": len(t["matches"]),
            "innings": len(t["innings"]),
            "balls_total": t["balls"],
            "runs_total": t["runs"],
            "sr_baseline": round(100.0 * own_rpb, 2),
            "pressure_balls": p["n"],
            "pressure_runs": p["runs"],
            "pressure_outs": p["outs"],
            "sr_pressure_actual": round(100.0 * p["runs"] / p["n"], 2),
            "sr_pressure_weighted": round(100.0 * pressure_rpb, 2),
            "expected_sr": round(100.0 * exp_rpb, 2),
            "weighted_balls": round(p["w"], 1),
            "ppi_raw": round(impact, 4),
            "ppi_shrunk": round(ppi, 4),
            "standard_error": round(se, 4),
            "effective_balls": round(n_eff, 1),
            "reliability": round(p["w"] / (p["w"] + CFG["shrink_k"]), 3),
            "pressure_innings": n_inn,
            "consistency": round(cons, 3) if cons is not None else "",
            "consistency_sd": round(sd, 4),
            "final_score": round(final, 4),
        })

    bowl_rows = []
    for pid, t in totals["bowl"].items():
        if any(tid in excluded_ids for tid in t["team"]):
            excluded_players.add(pid)
            continue
        if t["balls"] < CFG["min_total_balls_bowl"]:
            continue
        p = bowl_press.get(pid)
        if not p or p["n"] < CFG["min_pressure_balls_bowl"]:
            continue
        own_rpb = t["runs"] / t["balls"]
        own_wkt = t["wkts"] / t["balls"]
        pressure_rpb = p["wruns"] / p["w"]
        pressure_wkt = p["wwkts"] / p["w"]
        exp_rpb = p["wexp_runs"] / p["w"]
        exp_wkt = p["wexp_wkt"] / p["w"]
        base_rpb = CFG["absolute_weight"] * exp_rpb + (1 - CFG["absolute_weight"]) * own_rpb
        base_wkt = CFG["absolute_weight"] * exp_wkt + (1 - CFG["absolute_weight"]) * own_wkt
        impact, se, n_eff = weighted_impact_stats(
            p["samples"], base_rpb, base_wkt, CFG["wicket_value"], "bowl")
        ppi = impact * p["w"] / (p["w"] + CFG["shrink_k"])
        cons, n_inn, sd = consistency(p["inn"], own_rpb, own_wkt, "bowl")
        if n_inn < CFG["min_pressure_innings"]:
            continue
        final = ppi * (1.0 - CFG["consistency_weight"] + CFG["consistency_weight"] * cons)
        bowl_rows.append({
            "player_id": pid,
            "player_name": names.get(pid, pid),
            "team": team_name(t["team"]),
            "matches": len(t["matches"]),
            "innings": len(t["innings"]),
            "legal_balls": t["balls"],
            "runs_conceded": t["runs"],
            "economy_baseline": round(6.0 * own_rpb, 2),
            "wickets_total": t["wkts"],
            "pressure_balls": p["n"],
            "pressure_runs": p["runs"],
            "pressure_wickets": p["wkts"],
            "economy_pressure_actual": round(6.0 * p["runs"] / p["n"], 2),
            "economy_pressure_weighted": round(6.0 * pressure_rpb, 2),
            "expected_economy": round(6.0 * exp_rpb, 2),
            "weighted_balls": round(p["w"], 1),
            "ppi_raw": round(impact, 4),
            "ppi_shrunk": round(ppi, 4),
            "standard_error": round(se, 4),
            "effective_balls": round(n_eff, 1),
            "reliability": round(p["w"] / (p["w"] + CFG["shrink_k"]), 3),
            "pressure_innings": n_inn,
            "consistency": round(cons, 3) if cons is not None else "",
            "consistency_sd": round(sd, 4),
            "final_score": round(final, 4),
        })

    bat_rows.sort(key=lambda r: -r["final_score"])
    bowl_rows.sort(key=lambda r: -r["final_score"])
    for i, r in enumerate(bat_rows, 1):
        r["rank"] = i
    for i, r in enumerate(bowl_rows, 1):
        r["rank"] = i

    write_csv(os.path.join(OUT_DIR, "player_pressure_ratings_bat.csv"), bat_rows)
    write_csv(os.path.join(OUT_DIR, "player_pressure_ratings_bowl.csv"), bowl_rows)
    write_csv(os.path.join(OUT_DIR, "top10_batters.csv"), bat_rows[:10])
    write_csv(os.path.join(OUT_DIR, "top10_bowlers.csv"), bowl_rows[:10])

    with open(os.path.join(OUT_DIR, "ball_pressure.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["match_id", "innings", "striker_id", "bowler_id", "runs_off_bat",
                    "conceded", "faced", "legal", "batter_out", "bowler_wkt",
                    "pi_bat", "pi_bowl", "phase", "venue_factor"])
        for rec in records:
            w.writerow([rec[0], rec[1], rec[2], rec[3], rec[4], rec[5], int(rec[6]),
                        int(rec[7]), int(rec[8]), int(rec[9]),
                        round(rec[10], 2), round(rec[11], 2), rec[12], round(rec[13], 3)])

    summary = {
        "config": CFG,
        "pressure_thresholds": {k: round(v, 2) for k, v in thresholds.items()},
        "phase_par": {role: {str(p): [round(v[0], 4), round(v[1], 4)] for p, v in d.items()}
                      for role, d in phase_par.items()},
        "global_par_first_innings": round(global_par, 1),
        "eligible_batters": len(bat_rows),
        "eligible_bowlers": len(bowl_rows),
        "excluded_teams": CFG["exclude_teams"],
        "excluded_players": len(excluded_players),
        "matches": len(match_meta),
        "deliveries": len(records),
        "tiers": dict(Counter(m["tier"] for m in match_meta.values())),
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print("tiers:", summary["tiers"])
    print("pressure thresholds (top 30%):", summary["pressure_thresholds"])
    print("phase par (runs/ball, wkt/ball):", summary["phase_par"])
    print("excluded players (" + ", ".join(CFG["exclude_teams"]) + "):", len(excluded_players))
    print("eligible batters:", len(bat_rows), "| eligible bowlers:", len(bowl_rows))

    print("\nTOP 10 BATTERS")
    for r in bat_rows[:10]:
        print(f"{r['rank']:>2}. {r['player_name']:<28} {r['team']:<24} "
              f"score={r['final_score']:+.3f}  SR {r['sr_baseline']}->{r['sr_pressure_actual']} "
              f"(exp {r['expected_sr']})  balls={r['pressure_balls']}  inn={r['pressure_innings']}  cons={r['consistency']}")
    print("\nTOP 10 BOWLERS")
    for r in bowl_rows[:10]:
        print(f"{r['rank']:>2}. {r['player_name']:<28} {r['team']:<24} "
              f"score={r['final_score']:+.3f}  Econ {r['economy_baseline']}->{r['economy_pressure_actual']} "
              f"(exp {r['expected_economy']})  wkts={r['pressure_wickets']}  balls={r['pressure_balls']}  inn={r['pressure_innings']}  cons={r['consistency']}")
    return bat_rows, bowl_rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    run_pipeline()
