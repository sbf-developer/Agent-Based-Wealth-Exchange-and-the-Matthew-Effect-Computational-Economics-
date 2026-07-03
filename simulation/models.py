"""
Agent-based wealth-exchange models with identical initial endowments.

Core (no bias toward the already-wealthy):
  1. Dragulescu-Yakovenko (DY): fair random pairwise transfers.
  2. Chakraborti saving-propensity: agents retain a fraction lambda before trading.
  3. Yard-sale (Ispolatov-Krapivsky-Redner): fair coin, min-stake transfers.

Extension (Matthew effect / cumulative advantage):
  4. Biased exchange: transfers flow toward the richer agent with p > 1/2.

All models conserve total wealth. The headline result is that Models 1-3 produce
inequality from identical starts without rich-agent bias; Model 4 amplifies it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class SimulationConfig:
    n_agents: int = 1000
    initial_wealth: float = 1.0
    n_steps: int = 500_000
    seed: int = 42
    stake: float = 0.01  # DY transfer unit (fraction of mean wealth)
    saving_propensity: float = 0.9  # lambda in Chakraborti model
    yard_sale_fraction: float = 0.1  # fraction of loser's wealth transferred
    matthew_bias: float = 0.55  # P(wealth flows to richer agent)


def gini_coefficient(wealth: np.ndarray) -> float:
    """Gini index for non-negative wealth vector."""
    w = np.sort(np.asarray(wealth, dtype=float))
    w = w[w >= 0]
    if w.size == 0 or w.sum() == 0:
        return 0.0
    n = w.size
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * w) / (n * w.sum())) - (n + 1) / n)


def top_share(wealth: np.ndarray, fraction: float = 0.01) -> float:
    """Share of total wealth held by the top `fraction` of agents."""
    w = np.sort(np.asarray(wealth, dtype=float))[::-1]
    k = max(1, int(np.ceil(fraction * w.size)))
    return float(w[:k].sum() / w.sum())


def simulate(
    exchange_fn: Callable[[np.ndarray, np.random.Generator], None],
    config: SimulationConfig,
    record_every: int | None = None,
    batch_size: int = 500,
) -> dict:
    """
    Run a wealth-exchange simulation.

    Parameters
    ----------
    exchange_fn : callable
        In-place update for one micro-step: exchange_fn(wealth, rng).
    record_every : int, optional
        If set, store wealth snapshots and summary statistics at this interval.
    batch_size : int
        Number of pairwise meetings executed per outer step (vectorized).

    Returns
    -------
    dict with keys 'final_wealth', 'history' (optional), 'config'.
    """
    rng = np.random.default_rng(config.seed)
    wealth = np.full(config.n_agents, config.initial_wealth, dtype=float)
    total = config.n_agents * config.initial_wealth
    n_outer = max(1, config.n_steps // batch_size)

    history: list[dict] = []
    if record_every is not None:
        history.append(_snapshot(wealth, 0))

    micro = 0
    for outer in range(1, n_outer + 1):
        _batch_exchange(wealth, rng, exchange_fn, config, batch_size)
        micro = min(config.n_steps, outer * batch_size)

        s = wealth.sum()
        if s > 0 and abs(s - total) > 1e-6:
            wealth *= total / s

        if record_every and micro % record_every == 0:
            history.append(_snapshot(wealth, micro))

    if record_every and (config.n_steps % record_every != 0):
        history.append(_snapshot(wealth, config.n_steps))

    return {"final_wealth": wealth.copy(), "history": history, "config": config}


def _batch_exchange(
    wealth: np.ndarray,
    rng: np.random.Generator,
    exchange_fn: Callable,
    config: SimulationConfig,
    batch_size: int,
) -> None:
    """Vectorized batch of random pairwise exchanges."""
    n = wealth.size
    i = rng.integers(0, n, size=batch_size)
    j = rng.integers(0, n, size=batch_size)
    mask = i != j
    i, j = i[mask], j[mask]
    if i.size == 0:
        return

    model_fn = exchange_fn  # resolved by caller via partial application
    # Dispatch on model type using function name / attached metadata
    if hasattr(exchange_fn, "_model"):
        _vectorized_exchange(wealth, i, j, rng, exchange_fn._model, config)
    else:
        for k in range(min(batch_size, 64)):
            exchange_fn(wealth, rng)


def _vectorized_exchange(
    wealth: np.ndarray,
    i: np.ndarray,
    j: np.ndarray,
    rng: np.random.Generator,
    model: str,
    config: SimulationConfig,
) -> None:
    """Apply a batch of exchanges using wealth snapshot at batch start (Gillespie-style MC step)."""
    w0 = wealth.copy()
    delta_w = np.zeros_like(wealth)
    stake = config.stake * config.initial_wealth
    coin = rng.random(i.size) < 0.5

    if model == "dy":
        pay_ij = coin & (w0[i] >= stake)
        pay_ji = (~coin) & (w0[j] >= stake)
        np.add.at(delta_w, i, -pay_ij.astype(float) * stake + pay_ji.astype(float) * stake)
        np.add.at(delta_w, j, pay_ij.astype(float) * stake - pay_ji.astype(float) * stake)

    elif model == "chakraborti":
        lam = config.saving_propensity
        wi, wj = w0[i], w0[j]
        pool = (1.0 - lam) * (wi + wj) / 2.0
        new_i = np.maximum(lam * wi + np.where(coin, pool, -pool), 0.0)
        new_j = np.maximum(lam * wj + np.where(coin, -pool, pool), 0.0)
        np.add.at(delta_w, i, new_i - wi)
        np.add.at(delta_w, j, new_j - wj)

    elif model == "yard_sale":
        frac = config.yard_sale_fraction
        loser = np.where(coin, i, j)
        winner = np.where(coin, j, i)
        transfer = frac * np.minimum(w0[i], w0[j])
        np.add.at(delta_w, loser, -transfer)
        np.add.at(delta_w, winner, transfer)

    elif model == "matthew":
        bias = config.matthew_bias
        richer = np.where(w0[i] >= w0[j], i, j)
        poorer = np.where(w0[i] >= w0[j], j, i)
        transfer = np.minimum(stake, w0[poorer])
        to_rich = rng.random(i.size) < bias
        flow = np.where(to_rich, transfer, -transfer)
        np.add.at(delta_w, richer, flow)
        np.add.at(delta_w, poorer, -flow)

    wealth += delta_w
    np.maximum(wealth, 0.0, out=wealth)


def _snapshot(wealth: np.ndarray, step: int) -> dict:
    return {
        "step": step,
        "wealth": wealth.copy(),
        "gini": gini_coefficient(wealth),
        "top_1pct": top_share(wealth, 0.01),
        "top_10pct": top_share(wealth, 0.10),
        "median": float(np.median(wealth)),
        "mean": float(np.mean(wealth)),
    }


# --- Exchange rules (one micro-step = one random pair meeting) ---


def dy_exchange(wealth: np.ndarray, rng: np.random.Generator, stake: float) -> None:
    """Dragulescu-Yakovenko: fixed transfer; winner chosen with equal probability."""
    i, j = rng.integers(0, wealth.size, size=2)
    if i == j:
        return
    delta = stake
    if rng.random() < 0.5:
        if wealth[i] >= delta:
            wealth[i] -= delta
            wealth[j] += delta
    else:
        if wealth[j] >= delta:
            wealth[j] -= delta
            wealth[i] += delta


def chakraborti_exchange(
    wealth: np.ndarray, rng: np.random.Generator, lam: float
) -> None:
    """Saving-propensity model: each agent saves fraction lam before exchange."""
    i, j = rng.integers(0, wealth.size, size=2)
    if i == j:
        return
    wi, wj = wealth[i], wealth[j]
    d_w = (1.0 - lam) * (wi + wj) / 2.0
    if rng.random() < 0.5:
        wealth[i] = lam * wi + d_w
        wealth[j] = lam * wj - d_w
    else:
        wealth[i] = lam * wi - d_w
        wealth[j] = lam * wj + d_w
    wealth[i] = max(wealth[i], 0.0)
    wealth[j] = max(wealth[j], 0.0)


def yard_sale_exchange(
    wealth: np.ndarray, rng: np.random.Generator, frac: float
) -> None:
    """Yard-sale model: loser pays a fraction of min(w_i, w_j) to the winner."""
    i, j = rng.integers(0, wealth.size, size=2)
    if i == j:
        return
    if rng.random() < 0.5:
        loser, winner = i, j
    else:
        loser, winner = j, i
    delta = frac * min(wealth[i], wealth[j])
    if delta > 0:
        wealth[loser] -= delta
        wealth[winner] += delta


def matthew_exchange(
    wealth: np.ndarray, rng: np.random.Generator, bias: float, stake: float
) -> None:
    """
    Biased exchange: wealth flows to the richer agent with probability `bias` > 0.5.
    Captures cumulative advantage (Matthew effect) in a minimal ABM.
    """
    i, j = rng.integers(0, wealth.size, size=2)
    if i == j:
        return
    richer, poorer = (i, j) if wealth[i] >= wealth[j] else (j, i)
    delta = min(stake, wealth[poorer])
    if delta <= 0:
        return
    if rng.random() < bias:
        wealth[poorer] -= delta
        wealth[richer] += delta
    else:
        wealth[richer] -= delta
        wealth[poorer] += delta


def make_exchange_fn(model: str, config: SimulationConfig) -> Callable:
    """Factory for model-specific exchange functions."""
    if model == "dy":
        stake = config.stake * config.initial_wealth

        def fn(w, rng):
            dy_exchange(w, rng, stake)

        fn._model = "dy"
        return fn

    if model == "chakraborti":
        lam = config.saving_propensity

        def fn(w, rng):
            chakraborti_exchange(w, rng, lam)

        fn._model = "chakraborti"
        return fn

    if model == "yard_sale":
        frac = config.yard_sale_fraction

        def fn(w, rng):
            yard_sale_exchange(w, rng, frac)

        fn._model = "yard_sale"
        return fn

    if model == "matthew":
        stake = config.stake * config.initial_wealth
        bias = config.matthew_bias

        def fn(w, rng):
            matthew_exchange(w, rng, bias, stake)

        fn._model = "matthew"
        return fn

    raise ValueError(f"Unknown model: {model}")


UNBIASED_MODELS = ("dy", "chakraborti", "yard_sale")


def run_ensemble(
    model: str,
    config: SimulationConfig,
    n_runs: int = 1000,
) -> dict:
    """
    Run many independent simulations with distinct seeds.

    Each run starts from identical endowments (Gini = 0). Returns the distribution
    of terminal Gini coefficients across runs.
    """
    terminal_ginis: list[float] = []
    terminal_top1: list[float] = []

    for run in range(n_runs):
        cfg = replace(config, seed=config.seed + run)
        result = simulate(make_exchange_fn(model, cfg), cfg)
        w = result["final_wealth"]
        terminal_ginis.append(gini_coefficient(w))
        terminal_top1.append(top_share(w, 0.01))

    ginis = np.array(terminal_ginis)
    return {
        "model": model,
        "n_runs": n_runs,
        "ginis": ginis,
        "top_1pct": np.array(terminal_top1),
        "mean_gini": float(ginis.mean()),
        "median_gini": float(np.median(ginis)),
        "min_gini": float(ginis.min()),
        "max_gini": float(ginis.max()),
        "frac_gini_positive": float((ginis > 0).mean()),
        "config": config,
    }
