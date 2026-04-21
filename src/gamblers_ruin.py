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

    rng = np.random.default_rng(seed)
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
