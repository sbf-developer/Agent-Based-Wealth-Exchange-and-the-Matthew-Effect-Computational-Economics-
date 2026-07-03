"""
Generate publication-quality figures for the Matthew Effect paper.

Run from project root:
    python simulation/generate_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LogNorm

# Allow imports when run as script
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simulation.models import (  # noqa: E402
    SimulationConfig,
    gini_coefficient,
    make_exchange_fn,
    simulate,
    top_share,
)

FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Matplotlib style for academic figures
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

MODEL_LABELS = {
    "dy": "Random exchange (DY)",
    "chakraborti": "Saving propensity ($\\lambda=0.9$)",
    "yard_sale": "Yard-sale",
    "matthew": "Matthew effect (biased exchange)",
}

COLORS = {
    "dy": "#2166ac",
    "chakraborti": "#4393c3",
    "yard_sale": "#d6604d",
    "matthew": "#b2182b",
}


def base_config(**kwargs) -> SimulationConfig:
    defaults = dict(n_agents=1000, initial_wealth=1.0, n_steps=500_000, seed=42)
    defaults.update(kwargs)
    return SimulationConfig(**defaults)


def fig_gini_evolution():
    """Figure 1: Gini coefficient over time for all four models."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    record_every = 5000

    for model in ["dy", "chakraborti", "yard_sale", "matthew"]:
        cfg = base_config()
        if model == "chakraborti":
            cfg = base_config(saving_propensity=0.9)
        result = simulate(
            make_exchange_fn(model, cfg),
            cfg,
            record_every=record_every,
        )
        steps = [h["step"] for h in result["history"]]
        gini = [h["gini"] for h in result["history"]]
        ax.plot(
            steps,
            gini,
            label=MODEL_LABELS[model],
            color=COLORS[model],
            linewidth=2,
        )

    ax.set_xlabel("Interaction steps")
    ax.set_ylabel("Gini coefficient")
    ax.set_title("Emergent inequality from identical initial endowments")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", framealpha=0.95)
    fig.savefig(FIG_DIR / "fig_gini_evolution.pdf")
    fig.savefig(FIG_DIR / "fig_gini_evolution.png")
    plt.close(fig)
    print("Saved fig_gini_evolution")


def fig_wealth_distributions():
    """Figure 2: Final wealth distributions (log-log complementary CDF)."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    axes = axes.ravel()

    for ax, model in zip(axes, ["dy", "chakraborti", "yard_sale", "matthew"]):
        cfg = base_config()
        result = simulate(make_exchange_fn(model, cfg), cfg)
        w = np.sort(result["final_wealth"])
        w = w[w > 0]
        ranks = np.arange(1, len(w) + 1)
        ccdf = 1 - ranks / len(w)

        ax.loglog(w, ccdf + 1e-12, color=COLORS[model], linewidth=1.5)
        ax.set_xlabel("Wealth $w$")
        ax.set_ylabel("CCDF $P(W > w)$")
        ax.set_title(MODEL_LABELS[model])
        g = gini_coefficient(result["final_wealth"])
        ax.text(
            0.05,
            0.05,
            f"Gini = {g:.3f}",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    fig.suptitle(
        "Stationary wealth distributions after $5 \\times 10^5$ interactions",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_wealth_distributions.pdf")
    fig.savefig(FIG_DIR / "fig_wealth_distributions.png")
    plt.close(fig)
    print("Saved fig_wealth_distributions")


def fig_lorenz_curves():
    """Figure 3: Lorenz curves comparing models at terminal state."""
    fig, ax = plt.subplots(figsize=(6, 6))

    for model in ["dy", "chakraborti", "yard_sale", "matthew"]:
        cfg = base_config()
        result = simulate(make_exchange_fn(model, cfg), cfg)
        w = np.sort(result["final_wealth"])
        cum_pop = np.arange(1, len(w) + 1) / len(w)
        cum_wealth = np.cumsum(w) / w.sum()
        ax.plot(
            cum_pop,
            cum_wealth,
            label=f"{MODEL_LABELS[model]} (Gini={gini_coefficient(w):.2f})",
            color=COLORS[model],
            linewidth=2,
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect equality")
    ax.set_xlabel("Cumulative share of population (poorest to richest)")
    ax.set_ylabel("Cumulative share of wealth")
    ax.set_title("Lorenz curves after stochastic exchange")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_aspect("equal")
    fig.savefig(FIG_DIR / "fig_lorenz.pdf")
    fig.savefig(FIG_DIR / "fig_lorenz.png")
    plt.close(fig)
    print("Saved fig_lorenz")


def fig_heatmap_matthew():
    """Figure 4: Heatmap of terminal Gini vs bias and stake (Matthew model)."""
    biases = np.linspace(0.50, 0.65, 16)
    stakes = np.linspace(0.005, 0.05, 16)
    gini_grid = np.zeros((len(stakes), len(biases)))

    for i, stake in enumerate(stakes):
        for j, bias in enumerate(biases):
            cfg = base_config(
                n_steps=200_000,
                stake=stake,
                matthew_bias=bias,
                seed=123,
            )
            result = simulate(make_exchange_fn("matthew", cfg), cfg)
            gini_grid[i, j] = gini_coefficient(result["final_wealth"])

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        gini_grid,
        origin="lower",
        aspect="auto",
        cmap="YlOrRd",
        extent=[biases[0], biases[-1], stakes[0], stakes[-1]],
    )
    ax.set_xlabel("Bias toward richer agent ($p$)")
    ax.set_ylabel("Transfer stake ($\\Delta$)")
    ax.set_title("Terminal Gini in the Matthew-effect model")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Gini coefficient")
    fig.savefig(FIG_DIR / "fig_matthew_heatmap.pdf")
    fig.savefig(FIG_DIR / "fig_matthew_heatmap.png")
    plt.close(fig)
    print("Saved fig_matthew_heatmap")


def fig_top_shares():
    """Figure 5: Top 1% and top 10% wealth shares over time."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    record_every = 5000

    for model in ["dy", "yard_sale", "matthew"]:
        cfg = base_config()
        result = simulate(
            make_exchange_fn(model, cfg),
            cfg,
            record_every=record_every,
        )
        steps = [h["step"] for h in result["history"]]
        top1 = [h["top_1pct"] for h in result["history"]]
        ax.plot(
            steps,
            top1,
            label=f"{MODEL_LABELS[model]} (top 1\\%)",
            color=COLORS[model],
            linewidth=2,
        )

    ax.set_xlabel("Interaction steps")
    ax.set_ylabel("Wealth share of top 1\\%")
    ax.set_title("Wealth concentration dynamics")
    ax.legend(loc="lower right")
    fig.savefig(FIG_DIR / "fig_top_shares.pdf")
    fig.savefig(FIG_DIR / "fig_top_shares.png")
    plt.close(fig)
    print("Saved fig_top_shares")


def fig_equal_start_snapshot():
    """Figure 6: Wealth histogram evolution for Matthew model (3 snapshots)."""
    cfg = base_config()
    record_every = cfg.n_steps // 3
    result = simulate(
        make_exchange_fn("matthew", cfg),
        cfg,
        record_every=record_every,
    )

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
    for ax, snap in zip(axes, result["history"][:3]):
        w = snap["wealth"]
        ax.hist(w, bins=50, color=COLORS["matthew"], edgecolor="white", alpha=0.85)
        ax.set_xlabel("Wealth")
        ax.set_title(f"Step {snap['step']:,}\nGini = {snap['gini']:.3f}")
        if ax is axes[0]:
            ax.set_ylabel("Count")

    fig.suptitle(
        "From equality to concentration: Matthew-effect exchange",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_histogram_evolution.pdf")
    fig.savefig(FIG_DIR / "fig_histogram_evolution.png")
    plt.close(fig)
    print("Saved fig_histogram_evolution")


def fig_saving_propensity_sweep():
    """Figure 7: Terminal Gini vs saving propensity lambda."""
    lambdas = np.linspace(0.0, 0.99, 20)
    ginis = []

    for lam in lambdas:
        cfg = base_config(n_steps=300_000, saving_propensity=lam, seed=7)
        result = simulate(make_exchange_fn("chakraborti", cfg), cfg)
        ginis.append(gini_coefficient(result["final_wealth"]))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lambdas, ginis, "o-", color=COLORS["chakraborti"], linewidth=2, markersize=5)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Saving propensity $\\lambda$")
    ax.set_ylabel("Terminal Gini coefficient")
    ax.set_title("Inequality and saving in the Chakraborti model")
    fig.savefig(FIG_DIR / "fig_saving_sweep.pdf")
    fig.savefig(FIG_DIR / "fig_saving_sweep.png")
    plt.close(fig)
    print("Saved fig_saving_sweep")


def main():
    print(f"Writing figures to {FIG_DIR}")
    fig_gini_evolution()
    fig_wealth_distributions()
    fig_lorenz_curves()
    fig_heatmap_matthew()
    fig_top_shares()
    fig_equal_start_snapshot()
    fig_saving_propensity_sweep()
    print("All figures generated.")


if __name__ == "__main__":
    main()
