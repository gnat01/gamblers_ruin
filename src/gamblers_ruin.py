from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TrialResult:
    winner: int
    final_amounts: np.ndarray
    history: np.ndarray
    absorbed: bool
    rounds: int


@dataclass(frozen=True)
class ExperimentResult:
    wins: np.ndarray
    capped_trials: int
    total_rounds: int


@dataclass(frozen=True)
class SummaryRow:
    gambler: int
    initial: int
    expected: float
    outcomes: int
    observed: float


@dataclass(frozen=True)
class HurstRow:
    sample: int
    gambler: int
    window_start: int
    window_end: int
    window_mid: float
    hurst: float
    initial: int
    window_final: int
    absorbed: bool
    winner: int


@dataclass(frozen=True)
class SweepVector:
    vector_id: int
    family: str
    amounts: np.ndarray


def parse_amounts(raw: str) -> np.ndarray:
    amounts = np.array([int(part.strip()) for part in raw.split(",")], dtype=int)
    if amounts.ndim != 1 or len(amounts) < 2:
        raise ValueError("Provide at least two initial amounts.")
    if len(amounts) % 2 != 0:
        raise ValueError("The first setting assumes an even number of gamblers.")
    if np.any(amounts < 0):
        raise ValueError("Initial amounts must be nonnegative.")
    if np.count_nonzero(amounts) < 2:
        raise ValueError("At least two gamblers must start with positive amounts.")
    return amounts


def validate_amounts(amounts: np.ndarray) -> np.ndarray:
    if amounts.ndim != 1 or len(amounts) < 2:
        raise ValueError("Provide at least two initial amounts.")
    if len(amounts) % 2 != 0:
        raise ValueError("The first setting assumes an even number of gamblers.")
    if np.any(amounts < 0):
        raise ValueError("Initial amounts must be nonnegative.")
    if np.count_nonzero(amounts) < 2:
        raise ValueError("At least two gamblers must start with positive amounts.")
    return amounts


def split_total_by_weights(total: int, weights: np.ndarray) -> np.ndarray:
    raw = total * weights / weights.sum()
    amounts = np.floor(raw).astype(int)
    remainder = total - int(amounts.sum())
    if remainder:
        order = np.argsort(raw - amounts)[::-1]
        amounts[order[:remainder]] += 1
    return amounts


def generate_amounts(
    gamblers: int,
    total_wealth: int,
    mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if gamblers < 2:
        raise ValueError("--gamblers must be at least 2.")
    if gamblers % 2 != 0:
        raise ValueError("The first setting assumes an even number of gamblers.")
    if total_wealth < gamblers:
        raise ValueError("--total-wealth must be at least --gamblers so every gambler can start positive.")

    if mode == "equal":
        weights = np.ones(gamblers, dtype=float)
    elif mode == "descending":
        weights = np.arange(gamblers, 0, -1, dtype=float)
    elif mode == "random":
        weights = rng.random(gamblers)
    else:
        raise ValueError(f"Unknown amount mode: {mode}")

    amounts = split_total_by_weights(total_wealth - gamblers, weights) + 1
    return validate_amounts(amounts)


def get_initial_amounts(args: argparse.Namespace, rng: np.random.Generator) -> np.ndarray:
    if args.amounts is not None:
        return validate_amounts(parse_amounts(args.amounts))
    return generate_amounts(args.gamblers, args.total_wealth, args.amount_mode, rng)


def wealth_metrics(amounts: np.ndarray) -> dict[str, float]:
    total = float(amounts.sum())
    shares = amounts / total
    sorted_amounts = np.sort(amounts.astype(float))
    n = len(amounts)
    pairwise_sum = float((total * total - np.sum(amounts.astype(float) ** 2)) / 2)
    gini = float((2 * np.sum((np.arange(1, n + 1) * sorted_amounts))) / (n * total) - (n + 1) / n)
    positive_shares = shares[shares > 0]
    entropy = float(-np.sum(positive_shares * np.log(positive_shares)))
    hhi = float(np.sum(shares**2))
    return {
        "hhi": hhi,
        "one_minus_hhi": 1 - hhi,
        "pairwise_sum": pairwise_sum,
        "gini": gini,
        "max_share": float(shares.max()),
        "entropy": entropy,
    }


def amounts_from_weights(weights: np.ndarray, total_wealth: int) -> np.ndarray:
    if np.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError("Weights must be nonnegative and not all zero.")
    amounts = split_total_by_weights(total_wealth - len(weights), weights.astype(float)) + 1
    return np.sort(validate_amounts(amounts))[::-1]


def build_small_sweep_vectors(
    gamblers: int,
    total_wealth: int,
    vectors_per_family: int,
    rng: np.random.Generator,
) -> list[SweepVector]:
    if vectors_per_family < 1:
        raise ValueError("--sweep-vectors-per-family must be at least 1.")

    families: list[tuple[str, np.ndarray]] = []
    n = gamblers
    families.append(("equal", np.ones(n)))

    linear_powers = np.linspace(0.5, 3.0, vectors_per_family)
    for power in linear_powers:
        families.append((f"linear_descending_p{power:.2f}", np.arange(n, 0, -1, dtype=float) ** power))

    geometric_ratios = np.linspace(0.55, 0.9, vectors_per_family)
    for ratio in geometric_ratios:
        families.append((f"geometric_r{ratio:.2f}", ratio ** np.arange(n, dtype=float)))

    whale_shares = np.linspace(0.3, 0.75, vectors_per_family)
    for share in whale_shares:
        weights = np.full(n, (1 - share) / (n - 1))
        weights[0] = share
        families.append((f"one_whale_s{share:.2f}", weights))

    two_whale_shares = np.linspace(0.35, 0.75, vectors_per_family)
    for share in two_whale_shares:
        weights = np.full(n, (1 - share) / (n - 2))
        weights[0] = share * 0.58
        weights[1] = share * 0.42
        families.append((f"two_whales_s{share:.2f}", weights))

    dirichlet_alphas = np.geomspace(0.2, 5.0, vectors_per_family)
    for alpha in dirichlet_alphas:
        for draw in range(vectors_per_family):
            weights = rng.dirichlet(np.full(n, alpha))
            families.append((f"dirichlet_a{alpha:.2f}_d{draw}", weights))

    vectors = []
    seen: set[tuple[int, ...]] = set()
    for family, weights in families:
        amounts = amounts_from_weights(weights, total_wealth)
        key = tuple(int(value) for value in amounts)
        if key in seen:
            continue
        seen.add(key)
        vectors.append(SweepVector(vector_id=len(vectors), family=family, amounts=amounts))
    return vectors


def build_pairs(
    active: np.ndarray,
    amounts: np.ndarray,
    rng: np.random.Generator,
    *,
    pairing: str,
    warmup_rounds: int,
    round_index: int,
) -> np.ndarray:
    if pairing == "random" or round_index < warmup_rounds:
        ordered = active.copy()
        rng.shuffle(ordered)
    elif pairing == "ranked-after-warmup":
        tie_breaker = rng.random(len(active))
        order = np.lexsort((tie_breaker, -amounts[active]))
        ordered = active[order]
    else:
        raise ValueError(f"Unknown pairing strategy: {pairing}")

    paired_count = len(ordered) - (len(ordered) % 2)
    return ordered[:paired_count].reshape(-1, 2)


def simulate_trial(
    initial_amounts: Iterable[int],
    rng: np.random.Generator,
    *,
    max_rounds: int = 0,
    record_history: bool = True,
    pairing: str = "random",
    warmup_rounds: int = 0,
) -> TrialResult:
    amounts = np.array(initial_amounts, dtype=int)
    if len(amounts) % 2 != 0:
        raise ValueError("The first setting assumes an even number of gamblers.")
    if np.any(amounts < 0):
        raise ValueError("Initial amounts must be nonnegative.")

    history = [amounts.copy()] if record_history else []

    rounds = 0
    while max_rounds <= 0 or rounds < max_rounds:
        active = np.flatnonzero(amounts > 0)
        if len(active) <= 1:
            break

        pairs = build_pairs(
            active,
            amounts,
            rng,
            pairing=pairing,
            warmup_rounds=warmup_rounds,
            round_index=rounds,
        )

        first_wins = rng.integers(2, size=len(pairs), dtype=bool)
        winners = np.where(first_wins, pairs[:, 0], pairs[:, 1])
        losers = np.where(first_wins, pairs[:, 1], pairs[:, 0])
        np.add.at(amounts, winners, 1)
        np.add.at(amounts, losers, -1)

        if record_history:
            history.append(amounts.copy())
        rounds += 1

    active = np.flatnonzero(amounts > 0)
    absorbed = len(active) == 1
    winner = int(active[0]) if absorbed else int(np.argmax(amounts))
    history_array = np.array(history, dtype=int) if record_history else np.empty((0, len(amounts)), dtype=int)
    return TrialResult(
        winner=winner,
        final_amounts=amounts,
        history=history_array,
        absorbed=absorbed,
        rounds=rounds,
    )


def run_experiments(
    initial_amounts: np.ndarray,
    trials: int,
    seed: int | None,
    max_rounds: int,
    pairing: str,
    warmup_rounds: int,
) -> ExperimentResult:
    rng = np.random.default_rng(seed)
    wins = np.zeros(len(initial_amounts), dtype=int)
    capped_trials = 0
    total_rounds = 0

    for _ in range(trials):
        result = simulate_trial(
            initial_amounts,
            rng,
            max_rounds=max_rounds,
            record_history=False,
            pairing=pairing,
            warmup_rounds=warmup_rounds,
        )
        wins[result.winner] += 1
        capped_trials += int(not result.absorbed)
        total_rounds += result.rounds

    return ExperimentResult(wins=wins, capped_trials=capped_trials, total_rounds=total_rounds)


def build_summary_rows(initial_amounts: np.ndarray, result: ExperimentResult) -> list[SummaryRow]:
    wins = result.wins
    total = int(initial_amounts.sum())
    trials = int(wins.sum())
    expected = initial_amounts / total
    observed = wins / trials
    return [
        SummaryRow(
            gambler=i,
            initial=int(amount),
            expected=float(expected[i]),
            outcomes=int(wins[i]),
            observed=float(observed[i]),
        )
        for i, amount in enumerate(initial_amounts)
    ]


def print_experiment_summary(initial_amounts: np.ndarray, result: ExperimentResult) -> None:
    rows = build_summary_rows(initial_amounts, result)
    trials = int(result.wins.sum())
    richest = int(np.argmax(initial_amounts))

    print(f"initial amounts: {initial_amounts.tolist()}")
    print(f"total wealth: {int(initial_amounts.sum())}")
    print(f"trials: {trials}")
    print(f"absorbed trials: {trials - result.capped_trials}")
    print(f"capped trials: {result.capped_trials}")
    print(f"average rounds: {result.total_rounds / trials:,.1f}")
    print(f"richest gambler: {richest} with {initial_amounts[richest]}")
    print()
    outcome_label = "wins" if result.capped_trials == 0 else "wins/leader-at-cap"
    print(f"gambler  initial  expected  {outcome_label:>18}  observed")
    print(f"-------  -------  --------  {'-' * 18}  --------")
    for row in rows:
        print(f"{row.gambler:>7}  {row.initial:>7}  {row.expected:>8.2%}  {row.outcomes:>18}  {row.observed:>8.2%}")


def save_summary_table(
    path: Path,
    initial_amounts: np.ndarray,
    result: ExperimentResult,
    *,
    pairing: str,
    warmup_rounds: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = build_summary_rows(initial_amounts, result)
    trials = int(result.wins.sum())
    outcome_label = "wins" if result.capped_trials == 0 else "wins_or_leader_at_cap"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["initial_amounts", " ".join(str(int(amount)) for amount in initial_amounts)])
        writer.writerow(["total_wealth", int(initial_amounts.sum())])
        writer.writerow(["trials", trials])
        writer.writerow(["absorbed_trials", trials - result.capped_trials])
        writer.writerow(["capped_trials", result.capped_trials])
        writer.writerow(["average_rounds", result.total_rounds / trials])
        writer.writerow(["pairing", pairing])
        writer.writerow(["warmup_rounds", warmup_rounds])
        writer.writerow([])
        writer.writerow(["gambler", "initial", "expected_probability", outcome_label, "observed_probability"])
        for row in rows:
            writer.writerow([row.gambler, row.initial, row.expected, row.outcomes, row.observed])
    print(f"saved table to {path}")


def save_frequency_plot(path: Path, initial_amounts: np.ndarray, result: ExperimentResult) -> None:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    rows = build_summary_rows(initial_amounts, result)
    gamblers = np.array([row.gambler for row in rows])
    expected = np.array([row.expected for row in rows])
    observed = np.array([row.observed for row in rows])
    width = 0.4

    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(rows)), 5.5))
    ax.bar(gamblers - width / 2, expected, width=width, label="expected")
    ax.bar(gamblers + width / 2, observed, width=width, label="observed")
    ax.set_xlabel("gambler")
    ax.set_ylabel("survival frequency")
    ax.set_title("Expected vs observed survivor frequencies")
    ax.set_xticks(gamblers)
    ax.set_ylim(0, max(expected.max(), observed.max()) * 1.15 if len(rows) else 1)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"saved plot to {path}")


def estimate_hurst_rs(series: np.ndarray) -> float:
    values = np.asarray(series, dtype=float)
    if len(values) < 16 or np.all(values == values[0]):
        return float("nan")

    max_lag = len(values) // 2
    lags = np.unique(np.floor(np.logspace(np.log10(8), np.log10(max_lag), num=10)).astype(int))
    rs_values = []
    used_lags = []
    for lag in lags:
        if lag < 8:
            continue
        chunks = len(values) // lag
        if chunks < 2:
            continue
        chunk_rs = []
        for chunk in values[: chunks * lag].reshape(chunks, lag):
            centered = chunk - chunk.mean()
            cumulative = np.cumsum(centered)
            spread = cumulative.max() - cumulative.min()
            scale = chunk.std(ddof=1)
            if scale > 0:
                chunk_rs.append(spread / scale)
        if chunk_rs:
            used_lags.append(lag)
            rs_values.append(float(np.mean(chunk_rs)))

    if len(used_lags) < 2:
        return float("nan")
    slope, _ = np.polyfit(np.log(used_lags), np.log(rs_values), 1)
    return float(slope)


def collect_hurst_rows(
    initial_amounts: np.ndarray,
    *,
    seed: int | None,
    samples: int,
    max_rounds: int,
    pairing: str,
    warmup_rounds: int,
    window_size: int,
    step_size: int,
) -> tuple[list[HurstRow], list[TrialResult]]:
    rng = np.random.default_rng(None if seed is None else seed + 10_000)
    rows: list[HurstRow] = []
    trials: list[TrialResult] = []

    for sample in range(samples):
        result = simulate_trial(
            initial_amounts,
            rng,
            max_rounds=max_rounds,
            record_history=True,
            pairing=pairing,
            warmup_rounds=warmup_rounds,
        )
        trials.append(result)
        history = result.history
        if len(history) < window_size:
            continue

        for start in range(0, len(history) - window_size + 1, step_size):
            end = start + window_size
            window = history[start:end]
            for gambler in range(history.shape[1]):
                hurst = estimate_hurst_rs(window[:, gambler])
                if np.isnan(hurst):
                    continue
                rows.append(
                    HurstRow(
                        sample=sample,
                        gambler=gambler,
                        window_start=start,
                        window_end=end - 1,
                        window_mid=(start + end - 1) / 2,
                        hurst=hurst,
                        initial=int(initial_amounts[gambler]),
                        window_final=int(window[-1, gambler]),
                        absorbed=result.absorbed,
                        winner=result.winner,
                    )
                )

    return rows, trials


def save_hurst_table(path: Path, rows: list[HurstRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample",
                "gambler",
                "window_start",
                "window_end",
                "window_mid",
                "hurst",
                "initial",
                "window_final",
                "absorbed",
                "winner",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.sample,
                    row.gambler,
                    row.window_start,
                    row.window_end,
                    row.window_mid,
                    row.hurst,
                    row.initial,
                    row.window_final,
                    row.absorbed,
                    row.winner,
                ]
            )
    print(f"saved Hurst table to {path}")


def save_hurst_plot(path: Path, rows: list[HurstRow]) -> None:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No Hurst rows were available to plot. Try a smaller --hurst-window-size.")

    mids = np.array([row.window_mid for row in rows], dtype=float)
    hursts = np.array([row.hurst for row in rows], dtype=float)
    unique_mids = np.array(sorted(set(mids)))
    means = np.array([hursts[mids == mid].mean() for mid in unique_mids])
    p10 = np.array([np.percentile(hursts[mids == mid], 10) for mid in unique_mids])
    p90 = np.array([np.percentile(hursts[mids == mid], 90) for mid in unique_mids])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(unique_mids, means, linewidth=2, label="mean H")
    ax.fill_between(unique_mids, p10, p90, alpha=0.2, label="10th-90th percentile")
    ax.axhline(0.5, color="black", linewidth=1, linestyle="--", label="random-walk reference")
    ax.set_xlabel("round")
    ax.set_ylabel("windowed Hurst exponent")
    ax.set_title("Windowed Hurst exponent over sample trajectories")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"saved Hurst plot to {path}")


def save_trajectory_plot(path: Path, history: np.ndarray, title_suffix: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    rounds = np.arange(len(history))
    fig, ax = plt.subplots(figsize=(11, 6))
    for gambler in range(history.shape[1]):
        ax.plot(rounds, history[:, gambler], linewidth=1.6, label=f"gambler {gambler}")
    ax.set_xlabel("round")
    ax.set_ylabel("amount")
    ax.set_title(f"Wealth trajectories{title_suffix}")
    ax.grid(True, alpha=0.25)
    if history.shape[1] <= 12:
        ax.legend(loc="upper right", ncols=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"saved trajectory plot to {path}")


def percentile(values: list[int], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.array(values, dtype=float), q))


def run_small_absorption_sweep(
    *,
    output_path: Path,
    summary_path: Path | None,
    plot_path: Path | None,
    gamblers: int,
    total_wealth: int,
    sims_per_vector: int,
    vectors_per_family: int,
    seed: int | None,
    max_rounds: int,
    pairing: str,
    warmup_rounds: int,
) -> None:
    rng = np.random.default_rng(seed)
    vectors = build_small_sweep_vectors(gamblers, total_wealth, vectors_per_family, rng)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | int | float | bool]] = []
    with output_path.open("w", newline="") as handle:
        fieldnames = [
            "vector_id",
            "family",
            "amounts",
            "N",
            "S",
            "sim",
            "pairing",
            "warmup_rounds",
            "hhi",
            "one_minus_hhi",
            "pairwise_sum",
            "gini",
            "max_share",
            "entropy",
            "observed_time",
            "absorbed",
            "winner",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for vector in vectors:
            metrics = wealth_metrics(vector.amounts)
            for sim in range(sims_per_vector):
                result = simulate_trial(
                    vector.amounts,
                    rng,
                    max_rounds=max_rounds,
                    record_history=False,
                    pairing=pairing,
                    warmup_rounds=warmup_rounds,
                )
                row = {
                    "vector_id": vector.vector_id,
                    "family": vector.family,
                    "amounts": " ".join(str(int(amount)) for amount in vector.amounts),
                    "N": gamblers,
                    "S": total_wealth,
                    "sim": sim,
                    "pairing": pairing,
                    "warmup_rounds": warmup_rounds,
                    "hhi": metrics["hhi"],
                    "one_minus_hhi": metrics["one_minus_hhi"],
                    "pairwise_sum": metrics["pairwise_sum"],
                    "gini": metrics["gini"],
                    "max_share": metrics["max_share"],
                    "entropy": metrics["entropy"],
                    "observed_time": result.rounds,
                    "absorbed": result.absorbed,
                    "winner": result.winner,
                }
                writer.writerow(row)
                rows.append(row)

    print(f"saved small absorption sweep rows to {output_path}")
    print(f"sweep vectors: {len(vectors)}")
    print(f"simulations: {len(rows)}")

    summaries = summarize_sweep_rows(rows)
    if summary_path is not None:
        save_sweep_summary(summary_path, summaries)
    if plot_path is not None:
        save_sweep_plot(plot_path, summaries)


def summarize_sweep_rows(rows: list[dict[str, str | int | float | bool]]) -> list[dict[str, str | int | float]]:
    grouped: dict[int, list[dict[str, str | int | float | bool]]] = {}
    for row in rows:
        grouped.setdefault(int(row["vector_id"]), []).append(row)

    summaries: list[dict[str, str | int | float]] = []
    for vector_id, group in sorted(grouped.items()):
        times = [int(row["observed_time"]) for row in group]
        absorbed_times = [int(row["observed_time"]) for row in group if bool(row["absorbed"])]
        first = group[0]
        absorbed_count = len(absorbed_times)
        sims = len(group)
        summaries.append(
            {
                "vector_id": vector_id,
                "family": str(first["family"]),
                "amounts": str(first["amounts"]),
                "N": int(first["N"]),
                "S": int(first["S"]),
                "pairing": str(first["pairing"]),
                "warmup_rounds": int(first["warmup_rounds"]),
                "hhi": float(first["hhi"]),
                "one_minus_hhi": float(first["one_minus_hhi"]),
                "pairwise_sum": float(first["pairwise_sum"]),
                "gini": float(first["gini"]),
                "max_share": float(first["max_share"]),
                "entropy": float(first["entropy"]),
                "sims": sims,
                "absorbed_count": absorbed_count,
                "censored_count": sims - absorbed_count,
                "censored_fraction": (sims - absorbed_count) / sims,
                "mean_observed_time": float(np.mean(times)),
                "median_observed_time": float(np.median(times)),
                "p90_observed_time": percentile(times, 90),
                "p99_observed_time": percentile(times, 99),
                "mean_absorption_time_absorbed_only": float(np.mean(absorbed_times)) if absorbed_times else float("nan"),
                "median_absorption_time_absorbed_only": float(np.median(absorbed_times)) if absorbed_times else float("nan"),
                "p90_absorption_time_absorbed_only": percentile(absorbed_times, 90),
            }
        )
    return summaries


def save_sweep_summary(path: Path, summaries: list[dict[str, str | int | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not summaries:
        raise ValueError("No sweep summaries to save.")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"saved small absorption sweep summary to {path}")


def save_sweep_plot(path: Path, summaries: list[dict[str, str | int | float]]) -> None:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    if not summaries:
        raise ValueError("No sweep summaries to plot.")

    one_minus_hhi = np.array([float(row["one_minus_hhi"]) for row in summaries])
    pairwise = np.array([float(row["pairwise_sum"]) for row in summaries])
    median_t = np.array([float(row["median_absorption_time_absorbed_only"]) for row in summaries])
    mean_t = np.array([float(row["mean_absorption_time_absorbed_only"]) for row in summaries])
    censored = np.array([float(row["censored_fraction"]) for row in summaries])
    total_wealth = np.array([float(row["S"]) for row in summaries])

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].scatter(one_minus_hhi, median_t / (total_wealth**2), c=censored, cmap="viridis", edgecolor="black")
    axes[0, 0].set_xlabel("1 - HHI")
    axes[0, 0].set_ylabel("median T / S^2")
    axes[0, 0].set_title("Median absorption time")
    axes[0, 0].grid(True, alpha=0.25)

    axes[0, 1].scatter(one_minus_hhi, mean_t / (total_wealth**2), c=censored, cmap="viridis", edgecolor="black")
    axes[0, 1].set_xlabel("1 - HHI")
    axes[0, 1].set_ylabel("mean T / S^2")
    axes[0, 1].set_title("Mean absorption time")
    axes[0, 1].grid(True, alpha=0.25)

    axes[1, 0].scatter(pairwise, median_t, c=censored, cmap="viridis", edgecolor="black")
    axes[1, 0].set_xlabel("sum pairwise products")
    axes[1, 0].set_ylabel("median T")
    axes[1, 0].set_title("Pairwise product predictor")
    axes[1, 0].grid(True, alpha=0.25)

    scatter = axes[1, 1].scatter(one_minus_hhi, censored, c=pairwise, cmap="plasma", edgecolor="black")
    axes[1, 1].set_xlabel("1 - HHI")
    axes[1, 1].set_ylabel("censored fraction")
    axes[1, 1].set_title("Censoring check")
    axes[1, 1].grid(True, alpha=0.25)

    fig.colorbar(scatter, ax=axes[1, 1], label="pairwise product sum")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"saved small absorption sweep plot to {path}")


def animate_history(history: np.ndarray, output: Path | None, interval_ms: int) -> None:
    if output is not None:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(10, 6))
    rounds = np.arange(len(history))
    total = int(history[0].sum())
    lines = []

    for gambler in range(history.shape[1]):
        (line,) = ax.plot([], [], linewidth=2, label=f"gambler {gambler}")
        lines.append(line)

    ax.set_xlim(0, max(1, len(history) - 1))
    ax.set_ylim(0, total)
    ax.set_xlabel("round")
    ax.set_ylabel("amount")
    ax.set_title("Gambler's ruin with random pairings")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)

    def update(frame: int):
        visible_rounds = rounds[: frame + 1]
        for gambler, line in enumerate(lines):
            line.set_data(visible_rounds, history[: frame + 1, gambler])
        return lines

    animation = FuncAnimation(fig, update, frames=len(history), interval=interval_ms, blit=True)

    if output is None:
        plt.show()
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".gif":
        animation.save(output, writer="pillow")
    elif suffix in {".mp4", ".m4v"}:
        animation.save(output)
    else:
        raise ValueError("Animation output must end in .gif, .mp4, or .m4v.")
    print(f"saved animation to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Random-pairing gambler's ruin simulation.")
    parser.add_argument("--amounts", help="Comma-separated initial amounts. Overrides generated amounts.")
    parser.add_argument("--gamblers", type=int, default=4, help="Number of gamblers to generate when --amounts is omitted.")
    parser.add_argument("--total-wealth", type=int, default=100, help="Total initial wealth for generated amounts.")
    parser.add_argument(
        "--amount-mode",
        choices=("descending", "equal", "random"),
        default="descending",
        help="How to generate initial amounts when --amounts is omitted.",
    )
    parser.add_argument("--trials", "--sims", dest="trials", type=int, default=1_000, help="Number of simulations.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed. Use -1 for unpredictable randomness.")
    parser.add_argument("--max-rounds", type=int, default=0, help="Safety cap per trial. Use 0 for no cap.")
    parser.add_argument(
        "--pairing",
        choices=("random", "ranked-after-warmup"),
        default="random",
        help="Pairing policy for active gamblers.",
    )
    parser.add_argument(
        "--warmup-rounds",
        type=int,
        default=3,
        help="Random-pairing warmup rounds before ranked-after-warmup takes over.",
    )
    parser.add_argument("--animate", action="store_true", help="Animate one sample trajectory.")
    parser.add_argument("--output", type=Path, help="Optional animation output path: .gif, .mp4, or .m4v.")
    parser.add_argument("--save-table", type=Path, help="Save the summary table as a CSV file.")
    parser.add_argument("--save-plot", type=Path, help="Save expected vs observed survivor frequencies as an image.")
    parser.add_argument("--save-trajectory-plot", type=Path, help="Save a line plot for one sample wealth trajectory.")
    parser.add_argument("--save-hurst-table", type=Path, help="Save windowed Hurst estimates as a CSV file.")
    parser.add_argument("--save-hurst-plot", type=Path, help="Save a windowed Hurst summary plot.")
    parser.add_argument("--hurst-samples", type=int, default=3, help="Number of sample trajectories for Hurst analysis.")
    parser.add_argument("--hurst-window-size", type=int, default=256, help="Rounds per Hurst window.")
    parser.add_argument("--hurst-step-size", type=int, default=128, help="Round stride between Hurst windows.")
    parser.add_argument("--small-sweep-output", type=Path, help="Step-1 sweep: save one row per simulation to CSV.")
    parser.add_argument("--small-sweep-summary", type=Path, help="Step-1 sweep: save one aggregate row per initial vector.")
    parser.add_argument("--small-sweep-plot", type=Path, help="Step-1 sweep: save absorption-time predictor plots.")
    parser.add_argument("--sweep-vectors-per-family", type=int, default=4, help="Initial vectors per sweep family.")
    parser.add_argument("--sweep-sims-per-vector", type=int, default=50, help="Simulations per sweep initial vector.")
    parser.add_argument("--interval-ms", type=int, default=30, help="Animation frame interval.")
    args = parser.parse_args()

    seed = None if args.seed == -1 else args.seed
    if args.trials < 1:
        parser.error("--sims/--trials must be at least 1.")
    if args.warmup_rounds < 0:
        parser.error("--warmup-rounds must be nonnegative.")
    if args.hurst_samples < 1:
        parser.error("--hurst-samples must be at least 1.")
    if args.hurst_window_size < 16:
        parser.error("--hurst-window-size must be at least 16.")
    if args.hurst_step_size < 1:
        parser.error("--hurst-step-size must be at least 1.")
    if args.sweep_vectors_per_family < 1:
        parser.error("--sweep-vectors-per-family must be at least 1.")
    if args.sweep_sims_per_vector < 1:
        parser.error("--sweep-sims-per-vector must be at least 1.")

    rng = np.random.default_rng(seed)
    if args.small_sweep_output is not None:
        try:
            validate_amounts(np.ones(args.gamblers, dtype=int))
            if args.total_wealth < args.gamblers:
                raise ValueError("--total-wealth must be at least --gamblers so every gambler can start positive.")
            run_small_absorption_sweep(
                output_path=args.small_sweep_output,
                summary_path=args.small_sweep_summary,
                plot_path=args.small_sweep_plot,
                gamblers=args.gamblers,
                total_wealth=args.total_wealth,
                sims_per_vector=args.sweep_sims_per_vector,
                vectors_per_family=args.sweep_vectors_per_family,
                seed=seed,
                max_rounds=args.max_rounds,
                pairing=args.pairing,
                warmup_rounds=args.warmup_rounds,
            )
        except ValueError as exc:
            parser.error(str(exc))
        return

    try:
        amounts = get_initial_amounts(args, rng)
    except ValueError as exc:
        parser.error(str(exc))

    result = run_experiments(
        amounts,
        args.trials,
        seed,
        args.max_rounds,
        args.pairing,
        args.warmup_rounds,
    )
    print_experiment_summary(amounts, result)
    if args.save_table is not None:
        save_summary_table(
            args.save_table,
            amounts,
            result,
            pairing=args.pairing,
            warmup_rounds=args.warmup_rounds,
        )
    if args.save_plot is not None:
        save_frequency_plot(args.save_plot, amounts, result)

    needs_sample_histories = (
        args.save_hurst_table is not None
        or args.save_hurst_plot is not None
        or args.save_trajectory_plot is not None
    )
    sample_trials: list[TrialResult] = []
    if needs_sample_histories:
        hurst_rows, sample_trials = collect_hurst_rows(
            amounts,
            seed=seed,
            samples=args.hurst_samples,
            max_rounds=args.max_rounds,
            pairing=args.pairing,
            warmup_rounds=args.warmup_rounds,
            window_size=args.hurst_window_size,
            step_size=args.hurst_step_size,
        )
        if args.save_hurst_table is not None:
            save_hurst_table(args.save_hurst_table, hurst_rows)
        if args.save_hurst_plot is not None:
            if hurst_rows:
                save_hurst_plot(args.save_hurst_plot, hurst_rows)
            else:
                print("skipped Hurst plot: no valid Hurst windows; try a smaller --hurst-window-size")
        if args.save_trajectory_plot is not None and sample_trials:
            save_trajectory_plot(args.save_trajectory_plot, sample_trials[0].history, f" ({args.pairing})")

    if args.animate:
        if sample_trials:
            result = sample_trials[0]
        else:
            result = simulate_trial(
                amounts,
                rng,
                max_rounds=args.max_rounds,
                record_history=True,
                pairing=args.pairing,
                warmup_rounds=args.warmup_rounds,
            )
        print()
        outcome = "winner" if result.absorbed else "leader at cap"
        print(f"animated trial {outcome}: gambler {result.winner}")
        print(f"animated trial rounds: {result.rounds}")
        animate_history(result.history, args.output, args.interval_ms)


if __name__ == "__main__":
    main()
