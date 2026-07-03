"""Run a single simulation or Monte Carlo ensemble."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simulation.models import (  # noqa: E402
    SimulationConfig,
    gini_coefficient,
    make_exchange_fn,
    run_ensemble,
    simulate,
    top_share,
)


def main():
    parser = argparse.ArgumentParser(description="Run wealth-exchange ABM")
    parser.add_argument(
        "--model",
        choices=["dy", "chakraborti", "yard_sale", "matthew"],
        default="dy",
        help="dy = unbiased core model; matthew = biased extension",
    )
    parser.add_argument("--agents", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ensemble",
        type=int,
        default=0,
        metavar="N",
        help="If N > 0, run N independent simulations and summarize terminal Gini",
    )
    args = parser.parse_args()

    cfg = SimulationConfig(
        n_agents=args.agents,
        n_steps=args.steps,
        seed=args.seed,
    )

    if args.ensemble > 0:
        ens = run_ensemble(args.model, cfg, n_runs=args.ensemble)
        print(f"Model: {args.model} ({args.ensemble} runs)")
        print(f"Mean Gini: {ens['mean_gini']:.4f}")
        print(f"Median Gini: {ens['median_gini']:.4f}")
        print(f"Min / Max Gini: {ens['min_gini']:.4f} / {ens['max_gini']:.4f}")
        print(f"Fraction with G > 0: {ens['frac_gini_positive']:.4f}")
        return

    result = simulate(make_exchange_fn(args.model, cfg), cfg)
    w = result["final_wealth"]

    print(f"Model: {args.model}")
    print(f"Agents: {cfg.n_agents}, Steps: {cfg.n_steps}")
    print(f"Gini: {gini_coefficient(w):.4f}")
    print(f"Top 1% share: {top_share(w, 0.01):.4f}")
    print(f"Top 10% share: {top_share(w, 0.10):.4f}")
    print(
        f"Min / Median / Max wealth: {w.min():.4f} / "
        f"{float(np.median(w)):.4f} / {w.max():.4f}"
    )


if __name__ == "__main__":
    main()
