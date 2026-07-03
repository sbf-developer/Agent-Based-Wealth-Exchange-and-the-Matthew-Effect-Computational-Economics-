"""
Generate publication-quality figures for the Matthew Effect paper.

Narrative order: unbiased emergent inequality first, Matthew bias as extension.

Run from project root:
    python simulation/generate_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
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

FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

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
    "dy": "Random exchange (DY, unbiased)",
    "chakraborti": "Saving propensity ($\\lambda=0.9$, unbiased)",
    "yard_sale": "Yard-sale (unbiased)",
    "matthew": "Matthew effect ($p=0.55$, biased)",
}

COLORS = {
    "dy": "#2166ac",
    "chakraborti": "#4393c3",
    "yard_sale": "#d6604d",
    "matthew": "#b2182b",
    "unbiased": "#2166ac",
    "biased": "#b2182b",
}


def base_config(**kwargs) -> SimulationConfig:
    defaults = dict(n_agents=1000, initial_wealth=1.0, n_steps=500_000, seed=42)
    defaults.update(kwargs)
    return SimulationConfig(**defaults)


def _save(fig, name: str) -> None:
    fig.savefig(FIG_DIR / f"{name}.pdf")
    fig.savefig(FIG_DIR / f"{name}.png")
    plt.close(fig)
    print(f"Saved {name}")


def fig_ensemble_unbiased():
    """Centerpiece: 1000 unbiased DY runs, all starting from Gini = 0."""
    cfg = base_config()
    ens = run_ensemble("dy", cfg, n_runs=1000)
    ginis = ens["ginis"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(ginis, bins=40, color=COLORS["dy"], edgecolor="white", alpha=0.9)
    axes[0].axvline(0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Terminal Gini coefficient")
    axes[0].set_ylabel("Number of runs (out of 1000)")
    axes[0].set_title("Fair random exchange: 1000 independent runs")
    axes[0].text(
        0.97,
        0.95,
        f"All runs start at $G=0$\n"
        f"Mean $G={ens['mean_gini']:.3f}$\n"
        f"$\\min G={ens['min_gini']:.3f}$",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    sample_runs = [0, 100, 500, 999]
    for run_id in sample_runs:
        cfg_run = base_config(seed=cfg.seed + run_id)
        result = simulate(
            make_exchange_fn("dy", cfg_run),
            cfg_run,
            record_every=5000,
        )
        steps = [h["step"] for h in result["history"]]
        g = [h["gini"] for h in result["history"]]
        axes[1].plot(steps, g, linewidth=1.2, alpha=0.75, label=f"Run {run_id + 1}")

    axes[1].set_xlabel("Interaction steps")
    axes[1].set_ylabel("Gini coefficient")
    axes[1].set_title("Sample paths: same rules, different luck")
    axes[1].legend(fontsize=8, loc="lower right")

    fig.suptitle(
        "Emergent inequality without bias: every run starts equal",
        fontsize=13,
        y=1.03,
    )
    fig.tight_layout()
    _save(fig, "fig_ensemble_unbiased")


def fig_ensemble_compare():
    """Compare terminal Gini distributions: unbiased DY vs biased Matthew."""
    cfg = base_config()
    dy = run_ensemble("dy", cfg, n_runs=500)
    matthew = run_ensemble("matthew", cfg, n_runs=500)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, max(dy["ginis"].max(), matthew["ginis"].max()) * 1.05, 35)
    ax.hist(
        dy["ginis"],
        bins=bins,
        alpha=0.65,
        color=COLORS["dy"],
        label=f"Unbiased DY (mean $G$={dy['mean_gini']:.2f})",
        edgecolor="white",
    )
    ax.hist(
        matthew["ginis"],
        bins=bins,
        alpha=0.55,
        color=COLORS["matthew"],
        label=f"Matthew $p=0.55$ (mean $G$={matthew['mean_gini']:.2f})",
        edgecolor="white",
    )
    ax.set_xlabel("Terminal Gini coefficient")
    ax.set_ylabel("Number of runs (out of 500)")
    ax.set_title("Unbiased exchange vs.\\ Matthew-effect amplification")
    ax.legend(loc="upper right")
    _save(fig, "fig_ensemble_compare")


def fig_gini_evolution():
    """Gini over time: unbiased models first, Matthew last."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    record_every = 5000
    order = ["dy", "yard_sale", "chakraborti", "matthew"]

    for model in order:
        cfg = base_config()
        result = simulate(
            make_exchange_fn(model, cfg),
            cfg,
            record_every=record_every,
        )
        steps = [h["step"] for h in result["history"]]
        gini = [h["gini"] for h in result["history"]]
        lw = 2.5 if model == "dy" else 2
        zorder = 3 if model == "dy" else 2
        ax.plot(
            steps,
            gini,
            label=MODEL_LABELS[model],
            color=COLORS[model],
            linewidth=lw,
            zorder=zorder,
        )

    ax.set_xlabel("Interaction steps")
    ax.set_ylabel("Gini coefficient")
    ax.set_title("Inequality emerges from equal starts (DY needs no bias)")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=9)
    _save(fig, "fig_gini_evolution")


def fig_histogram_evolution_dy():
    """Wealth histograms at three snapshots under unbiased DY exchange."""
    cfg = base_config()
    record_every = cfg.n_steps // 3
    result = simulate(
        make_exchange_fn("dy", cfg),
        cfg,
        record_every=record_every,
    )

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
    for ax, snap in zip(axes, result["history"][:3]):
        w = snap["wealth"]
        ax.hist(w, bins=50, color=COLORS["dy"], edgecolor="white", alpha=0.85)
        ax.set_xlabel("Wealth")
        ax.set_title(f"Step {snap['step']:,}\nGini = {snap['gini']:.3f}")
        if ax is axes[0]:
            ax.set_ylabel("Count")

    fig.suptitle(
        "From equality to dispersion: fair random exchange (no rich-agent bias)",
        fontsize=13,
    )
    fig.tight_layout()
    _save(fig, "fig_histogram_evolution")


def fig_wealth_distributions():
    """Terminal CCDFs: unbiased models in top row, Matthew in bottom-right."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    axes = axes.ravel()
    models = ["dy", "yard_sale", "chakraborti", "matthew"]

    for ax, model in zip(axes, models):
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
        "Terminal wealth distributions after $5 \\times 10^5$ interactions",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig_wealth_distributions")


def fig_lorenz_curves():
    """Lorenz curves: unbiased models emphasized."""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    order = ["dy", "yard_sale", "chakraborti", "matthew"]

    for model in order:
        cfg = base_config()
        result = simulate(make_exchange_fn(model, cfg), cfg)
        w = np.sort(result["final_wealth"])
        cum_pop = np.arange(1, len(w) + 1) / len(w)
        cum_wealth = np.cumsum(w) / w.sum()
        lw = 2.5 if model == "dy" else 1.8
        ax.plot(
            cum_pop,
            cum_wealth,
            label=f"{MODEL_LABELS[model]} ($G={gini_coefficient(w):.2f}$)",
            color=COLORS[model],
            linewidth=lw,
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect equality")
    ax.set_xlabel("Cumulative share of population (poorest to richest)")
    ax.set_ylabel("Cumulative share of wealth")
    ax.set_title("Lorenz curves: unbiased rules already bend away from equality")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_aspect("equal")
    _save(fig, "fig_lorenz")


def fig_matthew_amplification():
    """Gini over time: unbiased DY vs Matthew bias (amplification story)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    record_every = 5000

    for model, label in [
        ("dy", "Unbiased random exchange"),
        ("matthew", "Matthew effect ($p=0.55$)"),
    ]:
        cfg = base_config()
        result = simulate(
            make_exchange_fn(model, cfg),
            cfg,
            record_every=record_every,
        )
        steps = [h["step"] for h in result["history"]]
        gini = [h["gini"] for h in result["history"]]
        ax.plot(steps, gini, label=label, color=COLORS[model], linewidth=2.5)

    ax.set_xlabel("Interaction steps")
    ax.set_ylabel("Gini coefficient")
    ax.set_title("Matthew effect as amplification, not prerequisite")
    ax.legend(loc="lower right")
    _save(fig, "fig_matthew_amplification")


def fig_heatmap_matthew():
    """Heatmap: terminal Gini vs bias p (extension model)."""
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
    ax.axvline(0.5, color="white", linestyle="--", linewidth=1.2, label="$p=1/2$ (no bias)")
    ax.set_xlabel("Bias toward richer agent ($p$)")
    ax.set_ylabel("Transfer stake ($\\Delta$)")
    ax.set_title("Extension: Matthew-effect model ($p > 1/2$)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Gini coefficient")
    _save(fig, "fig_matthew_heatmap")


def fig_saving_propensity_sweep():
    """Terminal Gini vs lambda (unbiased structural rule)."""
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
    ax.set_title("Stronger inequality from saving (still no rich-agent bias)")
    _save(fig, "fig_saving_sweep")


def main():
    print(f"Writing figures to {FIG_DIR}")
    fig_ensemble_unbiased()
    fig_ensemble_compare()
    fig_gini_evolution()
    fig_histogram_evolution_dy()
    fig_wealth_distributions()
    fig_lorenz_curves()
    fig_matthew_amplification()
    fig_heatmap_matthew()
    fig_saving_propensity_sweep()
    print("All figures generated.")


if __name__ == "__main__":
    main()
