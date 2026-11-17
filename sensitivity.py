#!/usr/bin/env python3
"""Sensitivity of the top-10 lists to the main modelling choices."""

import contextlib
import io
import os

import pipeline

BASE = {
    "wicket_value": pipeline.CFG["wicket_value"],
    "pressure_percentile": pipeline.CFG["pressure_percentile"],
    "absolute_weight": pipeline.CFG["absolute_weight"],
    "shrink_k": pipeline.CFG["shrink_k"],
    "soft_tau_frac": pipeline.CFG["soft_tau_frac"],
}


def run(overrides, tag):
    cfg = dict(BASE)
    cfg.update(overrides)
    pipeline.CFG.update(cfg)
    pipeline.OUT_DIR = os.path.join("outputs", "sensitivity", tag)
    with contextlib.redirect_stdout(io.StringIO()):
        bat, bowl = pipeline.run_pipeline()
    return ([r["player_name"] for r in bat[:10]], [r["player_name"] for r in bowl[:10]],
            {r["player_name"]: r["rank"] for r in bat}, {r["player_name"]: r["rank"] for r in bowl})


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


def main():
    base_bat, base_bowl, base_bat_rank, base_bowl_rank = run({}, "base")
    variants = {
        "wicket_value=2": {"wicket_value": 2.0},
        "wicket_value=8": {"wicket_value": 8.0},
        "pressure_top20%": {"pressure_percentile": 80},
        "pressure_top40%": {"pressure_percentile": 60},
        "absolute_weight=0.7": {"absolute_weight": 0.70},
        "absolute_weight=1.0": {"absolute_weight": 1.00},
        "soft_tau=0.05": {"soft_tau_frac": 0.05},
        "soft_tau=0.20": {"soft_tau_frac": 0.20},
    }
    print("base top10 batters:", base_bat)
    print("base top10 bowlers:", base_bowl)
    print()
    for tag, ov in variants.items():
        bat, bowl, bat_rank, bowl_rank = run(ov, tag.replace("=", "_").replace("%", "pct"))
        print(f"{tag:<24} bat overlap={jaccard(base_bat, bat):.2f}  bowl overlap={jaccard(base_bowl, bowl):.2f}")
        print("   batters:", bat)
        print("   bowlers:", bowl)


if __name__ == "__main__":
    main()
