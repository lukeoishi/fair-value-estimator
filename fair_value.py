"""Compare causal price estimators in a market with a known hidden value."""
from dataclasses import dataclass
from pathlib import Path
import argparse
import csv
import json
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Market:
    name: str
    process_std: float = 0.15
    quote_std: float = 1.0
    jump: float = 0.0
    gaps: bool = False


MARKETS = (
    Market("Calm market"),
    Market("Noisy quotes", quote_std=2.5),
    Market("Sudden jump", jump=5.0),
    Market("Missing quotes", gaps=True),
)


def simulate(market, n=600, seed=0):
    """Random-walk value plus independent quote noise; NaN means no quote."""
    if n < 10:
        raise ValueError("Use at least ten time steps.")
    rng = np.random.default_rng(seed)
    value = 100 + np.cumsum(rng.normal(0, market.process_std, n))
    value[n // 2:] += market.jump
    quotes = value + rng.normal(0, market.quote_std, n)
    if market.gaps:
        # Fixed gaps make the experiment easy to reproduce and inspect.
        for start in (n // 4, n // 2, 3 * n // 4):
            quotes[start:start + n // 20] = np.nan
    return value, quotes


def check_quotes(quotes):
    quotes = np.asarray(quotes, dtype=float)
    if quotes.ndim != 1 or not len(quotes) or not np.isfinite(quotes[0]):
        raise ValueError("Quotes must be a nonempty vector with a finite first quote.")
    if np.isinf(quotes).any():
        raise ValueError("Use NaN for missing quotes, not infinity.")
    return quotes


def latest_price(quotes):
    quotes = check_quotes(quotes)
    output = np.empty(len(quotes))
    current = quotes[0]
    for t, quote in enumerate(quotes):
        if np.isfinite(quote):
            current = quote
        output[t] = current
    return output


def ewma(quotes, alpha):
    quotes = check_quotes(quotes)
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1].")
    output = np.empty(len(quotes))
    current = quotes[0]
    for t, quote in enumerate(quotes):
        if np.isfinite(quote):
            current += alpha * (quote - current)
        output[t] = current
    return output


def kalman(quotes, process_variance, quote_variance):
    """Scalar random-walk filter. Returns estimates, variances and gains.

    Q: variance of one hidden-value step. R: variance of quote noise.
    Initial value is the first quote, with uncertainty R.
    """
    quotes = check_quotes(quotes)
    if not (np.isfinite(process_variance) and process_variance >= 0
            and np.isfinite(quote_variance) and quote_variance > 0):
        raise ValueError("Q must be finite and nonnegative; R finite and positive.")
    estimate = np.empty(len(quotes))
    variance = np.empty(len(quotes))
    gain = np.zeros(len(quotes))
    estimate[0], variance[0] = quotes[0], quote_variance
    for t in range(1, len(quotes)):
        predicted_variance = variance[t - 1] + process_variance
        estimate[t] = estimate[t - 1]
        variance[t] = predicted_variance
        if np.isfinite(quotes[t]):
            gain[t] = predicted_variance / (predicted_variance + quote_variance)
            estimate[t] += gain[t] * (quotes[t] - estimate[t])
            variance[t] = (1 - gain[t]) * predicted_variance
    return estimate, variance, gain


def rmse(estimate, value):
    return float(np.sqrt(np.mean((estimate - value) ** 2)))


def choose_alpha(market):
    """Tune only on separate synthetic calibration paths, never evaluation paths."""
    paths = [simulate(market, seed=seed) for seed in range(10)]
    candidates = np.linspace(0.01, 1.0, 100)
    scores = [np.mean([rmse(ewma(quotes, alpha), value)
                       for value, quotes in paths]) for alpha in candidates]
    return float(candidates[np.argmin(scores)])


def evaluate(market, alpha, seed):
    value, quotes = simulate(market, seed=seed)
    filtered, variance, gain = kalman(quotes, market.process_std ** 2,
                                      market.quote_std ** 2)
    estimates = {"Latest price": latest_price(quotes),
                 "EWMA": ewma(quotes, alpha), "Kalman": filtered}
    return value, quotes, estimates, variance, gain


def plot_examples(output, alphas):
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), constrained_layout=True)
    for ax, market in zip(axes, MARKETS):
        value, quotes, estimates, variance, _ = evaluate(market, alphas[market.name], 1000)
        t = np.arange(len(value))
        ax.scatter(t, quotes, s=4, color="#9ca3af", alpha=.55, label="Observed quote")
        ax.plot(t, value, color="#111827", lw=1.4, label="Hidden value")
        ax.plot(t, estimates["EWMA"], color="#d97706", lw=1.1, label="EWMA")
        ax.plot(t, estimates["Kalman"], color="#2563eb", lw=1.3, label="Kalman")
        radius = 1.96 * np.sqrt(variance)
        ax.fill_between(t, estimates["Kalman"] - radius, estimates["Kalman"] + radius,
                        color="#2563eb", alpha=.10, label="Model 95% interval")
        ax.set(title=market.name, ylabel="Price (arbitrary units)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.15)
    axes[0].legend(ncol=3, fontsize=8, loc="upper left")
    axes[-1].set_xlabel("Time step")
    fig.savefig(output / "examples.png", dpi=160)
    plt.close(fig)


def run(output, repeats):
    if repeats < 2:
        raise ValueError("Use at least two evaluation paths.")
    output.mkdir(parents=True, exist_ok=True)
    alphas = {market.name: choose_alpha(market) for market in MARKETS}
    rows = []
    for market in MARKETS:
        for seed in range(1000, 1000 + repeats):
            value, _, estimates, variance, _ = evaluate(market, alphas[market.name], seed)
            for name, estimate in estimates.items():
                rows.append({"scenario": market.name, "seed": seed, "estimator": name,
                             "rmse": rmse(estimate, value),
                             "mae": float(np.mean(np.abs(estimate - value))),
                             "interval_coverage": float(np.mean(np.abs(estimate - value)
                                 <= 1.96 * np.sqrt(variance))) if name == "Kalman" else ""})
    with (output / "runs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Evaluation", "", f"Mean RMSE across {repeats} evaluation paths per scenario. Lower is better.", "",
             "| Scenario | Latest price | EWMA | Kalman | EWMA alpha |",
             "|---|---:|---:|---:|---:|"]
    for market in MARKETS:
        scores = [np.mean([r["rmse"] for r in rows if r["scenario"] == market.name
                          and r["estimator"] == name]) for name in ("Latest price", "EWMA", "Kalman")]
        lines.append(f"| {market.name} | " + " | ".join(f"{x:.3f}" for x in scores)
                     + f" | {alphas[market.name]:.2f} |")
    (output / "results.md").write_text("\n".join(lines) + "\n")
    (output / "config.json").write_text(json.dumps({"python": platform.python_version(),
        "numpy": np.__version__, "matplotlib": matplotlib.__version__, "steps": 600,
        "calibration_seeds": [0, 9], "evaluation_seeds": [1000, 999 + repeats],
        "ewma_alpha": alphas, "markets": [vars(m) for m in MARKETS]}, indent=2) + "\n")
    plot_examples(output, alphas)
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    run(args.output, args.repeats)
