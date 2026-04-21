from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TrialResult:
    winner: int
    final_amounts: np.ndarray
    history: np.ndarray


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


def simulate_trial(
    initial_amounts: Iterable[int],
    rng: np.random.Generator,
    *,
    max_rounds: int = 1_000_000,
    record_history: bool = True,
) -> TrialResult:
    amounts = np.array(initial_amounts, dtype=int)
    if len(amounts) % 2 != 0:
        raise ValueError("The first setting assumes an even number of gamblers.")
    if np.any(amounts < 0):
        raise ValueError("Initial amounts must be nonnegative.")

    history = [amounts.copy()] if record_history else []

    for _ in range(max_rounds):
        active = np.flatnonzero(amounts > 0)
        if len(active) <= 1:
            break

        rng.shuffle(active)
        paired_count = len(active) - (len(active) % 2)
        pairs = active[:paired_count].reshape(-1, 2)

        first_wins = rng.integers(2, size=len(pairs), dtype=bool)
        winners = np.where(first_wins, pairs[:, 0], pairs[:, 1])
        losers = np.where(first_wins, pairs[:, 1], pairs[:, 0])
        np.add.at(amounts, winners, 1)
        np.add.at(amounts, losers, -1)

        if record_history:
            history.append(amounts.copy())
    else:
        raise RuntimeError(f"Trial did not finish within {max_rounds:,} rounds.")

    active = np.flatnonzero(amounts > 0)
    winner = int(active[0]) if len(active) == 1 else int(np.argmax(amounts))
    history_array = np.array(history, dtype=int) if record_history else np.empty((0, len(amounts)), dtype=int)
    return TrialResult(winner=winner, final_amounts=amounts, history=history_array)


def run_experiments(
    initial_amounts: np.ndarray,
    trials: int,
    seed: int | None,
    max_rounds: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    wins = np.zeros(len(initial_amounts), dtype=int)

    for _ in range(trials):
        result = simulate_trial(initial_amounts, rng, max_rounds=max_rounds, record_history=False)
        wins[result.winner] += 1

    return wins


def print_experiment_summary(initial_amounts: np.ndarray, wins: np.ndarray) -> None:
    total = int(initial_amounts.sum())
    trials = int(wins.sum())
    expected = initial_amounts / total
    observed = wins / trials
    richest = int(np.argmax(initial_amounts))

    print(f"initial amounts: {initial_amounts.tolist()}")
    print(f"total wealth: {total}")
    print(f"trials: {trials}")
    print(f"richest gambler: {richest} with {initial_amounts[richest]}")
    print()
    print("gambler  initial  expected    wins  observed")
    print("-------  -------  --------  ------  --------")
    for i, amount in enumerate(initial_amounts):
        print(f"{i:>7}  {amount:>7}  {expected[i]:>8.2%}  {wins[i]:>6}  {observed[i]:>8.2%}")


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
    parser.add_argument("--amounts", default="50,25,15,10", help="Comma-separated initial amounts.")
    parser.add_argument("--trials", type=int, default=1_000, help="Number of repeated experiments.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed. Use -1 for unpredictable randomness.")
    parser.add_argument("--max-rounds", type=int, default=1_000_000, help="Safety cap per trial.")
    parser.add_argument("--animate", action="store_true", help="Animate one sample trajectory.")
    parser.add_argument("--output", type=Path, help="Optional animation output path: .gif, .mp4, or .m4v.")
    parser.add_argument("--interval-ms", type=int, default=30, help="Animation frame interval.")
    args = parser.parse_args()

    amounts = parse_amounts(args.amounts)
    seed = None if args.seed == -1 else args.seed

    wins = run_experiments(amounts, args.trials, seed, args.max_rounds)
    print_experiment_summary(amounts, wins)

    if args.animate:
        rng = np.random.default_rng(seed)
        result = simulate_trial(amounts, rng, max_rounds=args.max_rounds, record_history=True)
        print()
        print(f"animated trial winner: gambler {result.winner}")
        print(f"rounds until absorption: {len(result.history) - 1}")
        animate_history(result.history, args.output, args.interval_ms)


if __name__ == "__main__":
    main()
