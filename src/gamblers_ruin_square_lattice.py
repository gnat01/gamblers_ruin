from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


NEIGHBOR_OFFSETS = {
    "neumann": [(1, 0), (-1, 0), (0, 1), (0, -1)],
    "moore": [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)],
}


@dataclass(frozen=True)
class LatticeResult:
    initial: np.ndarray
    final: np.ndarray
    frames: list[np.ndarray]
    metrics: list[dict[str, float | int]]
    frozen: bool
    rounds: int


def initial_lattice(
    side: int,
    mode: str,
    initial_wealth: int,
    total_wealth: int | None,
    heterogeneity: float,
    seed_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if side < 2:
        raise ValueError("--N must be at least 2.")
    if initial_wealth < 1:
        raise ValueError("--initial-wealth must be at least 1.")
    sites = side * side
    total = total_wealth if total_wealth is not None else sites * initial_wealth
    if total < sites:
        raise ValueError("--total-wealth must be at least N*N so every site can start positive.")

    if mode == "uniform":
        weights = np.ones(sites)
    elif mode == "random-gamma":
        if heterogeneity <= 0:
            raise ValueError("--heterogeneity must be positive for random-gamma.")
        weights = rng.gamma(shape=heterogeneity, scale=1.0, size=sites)
    elif mode == "gradient":
        x = np.linspace(0.25, 1.75, side)
        weights = np.tile(x, (side, 1)).ravel()
    elif mode == "seeds":
        if seed_count < 1:
            raise ValueError("--seed-count must be at least 1.")
        weights = np.full(sites, 0.2)
        seed_indices = rng.choice(sites, size=min(seed_count, sites), replace=False)
        weights[seed_indices] = max(1.0, heterogeneity)
    else:
        raise ValueError(f"Unknown initial mode: {mode}")

    raw = (total - sites) * weights / weights.sum()
    amounts = np.floor(raw).astype(int) + 1
    remainder = total - int(amounts.sum())
    if remainder:
        fractions = raw - np.floor(raw)
        order = np.argsort(fractions)[::-1]
        amounts[order[:remainder]] += 1
    return amounts.reshape(side, side)


def neighbor_edges(side: int, neighborhood: str) -> np.ndarray:
    offsets = NEIGHBOR_OFFSETS[neighborhood]
    edges = []
    for r in range(side):
        for c in range(side):
            a = r * side + c
            for dr, dc in offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < side and 0 <= nc < side:
                    b = nr * side + nc
                    if a < b:
                        edges.append((a, b))
    return np.array(edges, dtype=int)


def active_edges(amounts: np.ndarray, edges: np.ndarray) -> np.ndarray:
    flat = amounts.ravel()
    mask = (flat[edges[:, 0]] > 0) & (flat[edges[:, 1]] > 0)
    return edges[mask]


def random_matching(active_edge_array: np.ndarray, site_count: int, rng: np.random.Generator) -> np.ndarray:
    if len(active_edge_array) == 0:
        return active_edge_array
    order = rng.permutation(len(active_edge_array))
    used = np.zeros(site_count, dtype=bool)
    matched = []
    for idx in order:
        a, b = active_edge_array[idx]
        if not used[a] and not used[b]:
            used[a] = True
            used[b] = True
            matched.append((a, b))
    return np.array(matched, dtype=int) if matched else np.empty((0, 2), dtype=int)


def component_sizes(active: np.ndarray, neighborhood: str) -> list[int]:
    side = active.shape[0]
    seen = np.zeros_like(active, dtype=bool)
    sizes = []
    offsets = NEIGHBOR_OFFSETS[neighborhood]
    for r in range(side):
        for c in range(side):
            if not active[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            size = 0
            while stack:
                cr, cc = stack.pop()
                size += 1
                for dr, dc in offsets:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < side and 0 <= nc < side and active[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            sizes.append(size)
    return sizes


def interface_length(active: np.ndarray, neighborhood: str) -> int:
    side = active.shape[0]
    length = 0
    for dr, dc in NEIGHBOR_OFFSETS[neighborhood]:
        if dr < 0 or (dr == 0 and dc < 0):
            continue
        for r in range(side):
            for c in range(side):
                nr, nc = r + dr, c + dc
                if 0 <= nr < side and 0 <= nc < side and active[r, c] != active[nr, nc]:
                    length += 1
    return length


def morans_i(values: np.ndarray, neighborhood: str) -> float:
    side = values.shape[0]
    x = values.astype(float)
    mean = x.mean()
    centered = x - mean
    denom = float(np.sum(centered**2))
    if denom == 0:
        return float("nan")
    numerator = 0.0
    weights = 0
    for r in range(side):
        for c in range(side):
            for dr, dc in NEIGHBOR_OFFSETS[neighborhood]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < side and 0 <= nc < side:
                    numerator += centered[r, c] * centered[nr, nc]
                    weights += 1
    return float((side * side / weights) * (numerator / denom)) if weights else float("nan")


def lattice_metrics(round_index: int, amounts: np.ndarray, neighborhood: str) -> dict[str, float | int]:
    active = amounts > 0
    sizes = component_sizes(active, neighborhood)
    active_count = int(active.sum())
    total_sites = amounts.size
    largest = max(sizes) if sizes else 0
    positive = amounts[active]
    return {
        "round": round_index,
        "active_sites": active_count,
        "active_density": active_count / total_sites,
        "cluster_count": len(sizes),
        "largest_cluster": largest,
        "largest_cluster_fraction": largest / active_count if active_count else 0.0,
        "interface_length": interface_length(active, neighborhood),
        "total_wealth": int(amounts.sum()),
        "max_wealth": int(amounts.max()),
        "mean_active_wealth": float(positive.mean()) if active_count else 0.0,
        "moran_i_wealth": morans_i(amounts, neighborhood),
    }


def simulate_lattice(
    initial: np.ndarray,
    neighborhood: str,
    max_rounds: int,
    rng: np.random.Generator,
    frame_every: int,
    metric_every: int,
) -> LatticeResult:
    amounts = initial.copy()
    side = amounts.shape[0]
    edges = neighbor_edges(side, neighborhood)
    frames = [amounts.copy()]
    metrics = [lattice_metrics(0, amounts, neighborhood)]

    frozen = False
    rounds = 0
    while max_rounds <= 0 or rounds < max_rounds:
        current_active_edges = active_edges(amounts, edges)
        if len(current_active_edges) == 0:
            frozen = True
            break

        matching = random_matching(current_active_edges, side * side, rng)
        flat = amounts.ravel()
        first_wins = rng.integers(2, size=len(matching), dtype=bool)
        winners = np.where(first_wins, matching[:, 0], matching[:, 1])
        losers = np.where(first_wins, matching[:, 1], matching[:, 0])
        np.add.at(flat, winners, 1)
        np.add.at(flat, losers, -1)
        rounds += 1

        if frame_every > 0 and rounds % frame_every == 0:
            frames.append(amounts.copy())
        if metric_every > 0 and rounds % metric_every == 0:
            metrics.append(lattice_metrics(rounds, amounts, neighborhood))

    if len(frames) == 0 or not np.array_equal(frames[-1], amounts):
        frames.append(amounts.copy())
    if int(metrics[-1]["round"]) != rounds:
        metrics.append(lattice_metrics(rounds, amounts, neighborhood))
    return LatticeResult(initial=initial, final=amounts, frames=frames, metrics=metrics, frozen=frozen, rounds=rounds)


def save_metrics(path: Path, metrics: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)
    print(f"saved metrics to {path}")


def save_final_summary(path: Path, result: LatticeResult, neighborhood: str) -> None:
    metrics = lattice_metrics(result.rounds, result.final, neighborhood)
    active = result.final > 0
    sizes = sorted(component_sizes(active, neighborhood), reverse=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["rounds", result.rounds])
        writer.writerow(["frozen", result.frozen])
        for key, value in metrics.items():
            writer.writerow([key, value])
        writer.writerow(["cluster_sizes", " ".join(str(size) for size in sizes)])
    print(f"saved final summary to {path}")


def save_heatmaps(path: Path, result: LatticeResult, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    vmax = max(int(result.initial.max()), int(result.final.max()))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    panels = [(result.initial, "initial wealth"), (result.final, "final wealth"), (result.final > 0, "final active islands")]
    for ax, (data, label) in zip(axes, panels):
        if data.dtype == bool:
            image = ax.imshow(data, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        else:
            image = ax.imshow(data, cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
        if data.dtype != bool:
            fig.colorbar(image, ax=ax, fraction=0.046)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f"saved heatmaps to {path}")


def save_metrics_plot(path: Path, metrics: list[dict[str, float | int]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    rounds = np.array([row["round"] for row in metrics], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    series = [
        ("active_density", "active density"),
        ("cluster_count", "cluster count"),
        ("largest_cluster_fraction", "largest cluster / active sites"),
        ("interface_length", "interface length"),
    ]
    for ax, (key, label) in zip(axes.ravel(), series):
        ax.plot(rounds, [row[key] for row in metrics], linewidth=2)
        ax.set_xlabel("round")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f"saved metrics plot to {path}")


def save_animation(path: Path, frames: list[np.ndarray], interval_ms: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    path.parent.mkdir(parents=True, exist_ok=True)
    vmax = max(int(frame.max()) for frame in frames)
    fig, ax = plt.subplots(figsize=(6, 6))
    image = ax.imshow(frames[0], cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.046)

    def update(frame_index: int):
        image.set_data(frames[frame_index])
        ax.set_title(f"frame {frame_index + 1}/{len(frames)}")
        return [image]

    animation = FuncAnimation(fig, update, frames=len(frames), interval=interval_ms, blit=True)
    if path.suffix.lower() == ".gif":
        animation.save(path, writer="pillow")
    else:
        animation.save(path)
    plt.close(fig)
    print(f"saved animation to {path}")


def run_bifurcation(args: argparse.Namespace, seed: int | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    values = np.linspace(args.bifurcation_min, args.bifurcation_max, args.bifurcation_steps)
    rows = []
    for heterogeneity in values:
        for sim in range(args.sims):
            initial = initial_lattice(
                args.N,
                "random-gamma",
                args.initial_wealth,
                args.total_wealth,
                heterogeneity,
                args.seed_count,
                rng,
            )
            result = simulate_lattice(
                initial,
                args.neighborhood,
                args.max_rounds,
                rng,
                frame_every=0,
                metric_every=max(1, args.metric_every),
            )
            final_metrics = lattice_metrics(result.rounds, result.final, args.neighborhood)
            rows.append(
                {
                    "heterogeneity": heterogeneity,
                    "sim": sim,
                    "rounds": result.rounds,
                    "frozen": result.frozen,
                    "active_density": final_metrics["active_density"],
                    "cluster_count": final_metrics["cluster_count"],
                    "largest_cluster_fraction": final_metrics["largest_cluster_fraction"],
                    "interface_length": final_metrics["interface_length"],
                    "max_wealth": final_metrics["max_wealth"],
                }
            )

    if args.bifurcation_output is not None:
        args.bifurcation_output.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
        specs = [
            ("active_density", "final active density"),
            ("cluster_count", "final cluster count"),
            ("largest_cluster_fraction", "largest island fraction"),
            ("rounds", "rounds to freeze/cap"),
        ]
        for ax, (key, label) in zip(axes.ravel(), specs):
            ax.scatter([row["heterogeneity"] for row in rows], [row[key] for row in rows], s=18, alpha=0.7)
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.25)
        for ax in axes[-1]:
            ax.set_xlabel("random-gamma heterogeneity alpha")
        fig.suptitle(f"Lattice ruin bifurcation scan ({args.neighborhood})")
        fig.tight_layout()
        fig.savefig(args.bifurcation_output, dpi=170)
        plt.close(fig)
        print(f"saved bifurcation plot to {args.bifurcation_output}")

    if args.bifurcation_table is not None:
        args.bifurcation_table.parent.mkdir(parents=True, exist_ok=True)
        with args.bifurcation_table.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved bifurcation table to {args.bifurcation_table}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Square-lattice multiplayer gambler's ruin.")
    parser.add_argument("--N", type=int, default=40, help="Square side length.")
    parser.add_argument("--neighborhood", choices=("neumann", "moore"), default="neumann")
    parser.add_argument("--sims", "--trials", dest="sims", type=int, default=1, help="Number of simulations.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed. Use -1 for unpredictable randomness.")
    parser.add_argument("--max-rounds", type=int, default=0, help="Round cap. Use 0 to run until frozen.")
    parser.add_argument("--initial-mode", choices=("uniform", "random-gamma", "gradient", "seeds"), default="uniform")
    parser.add_argument("--initial-wealth", type=int, default=5, help="Default wealth scale per site.")
    parser.add_argument("--total-wealth", type=int, help="Optional exact total initial wealth.")
    parser.add_argument("--heterogeneity", type=float, default=1.0, help="Gamma alpha or seed strength.")
    parser.add_argument("--seed-count", type=int, default=5, help="Number of rich seeds for --initial-mode seeds.")
    parser.add_argument("--frame-every", type=int, default=10, help="Save one animation frame every this many rounds.")
    parser.add_argument("--metric-every", type=int, default=10, help="Save one metric row every this many rounds.")
    parser.add_argument("--save-animation", type=Path, help="Save lattice wealth animation, usually .gif.")
    parser.add_argument("--save-heatmaps", type=Path, help="Save initial/final heatmaps and final active mask.")
    parser.add_argument("--save-metrics", type=Path, help="Save time-series pattern metrics as CSV.")
    parser.add_argument("--save-metrics-plot", type=Path, help="Save time-series pattern metrics plot.")
    parser.add_argument("--save-final-summary", type=Path, help="Save final frozen-island summary CSV.")
    parser.add_argument("--interval-ms", type=int, default=80, help="Animation frame interval.")
    parser.add_argument("--bifurcation-output", type=Path, help="Save heterogeneity scan plot.")
    parser.add_argument("--bifurcation-table", type=Path, help="Save heterogeneity scan raw CSV.")
    parser.add_argument("--bifurcation-min", type=float, default=0.2)
    parser.add_argument("--bifurcation-max", type=float, default=5.0)
    parser.add_argument("--bifurcation-steps", type=int, default=12)
    args = parser.parse_args()

    if args.sims < 1:
        parser.error("--sims must be at least 1.")
    if args.frame_every < 0 or args.metric_every < 1:
        parser.error("--frame-every must be nonnegative and --metric-every must be positive.")
    if args.bifurcation_steps < 2:
        parser.error("--bifurcation-steps must be at least 2.")

    seed = None if args.seed == -1 else args.seed
    if args.bifurcation_output is not None or args.bifurcation_table is not None:
        run_bifurcation(args, seed)
        return

    rng = np.random.default_rng(seed)
    result = None
    for sim in range(args.sims):
        initial = initial_lattice(
            args.N,
            args.initial_mode,
            args.initial_wealth,
            args.total_wealth,
            args.heterogeneity,
            args.seed_count,
            rng,
        )
        result = simulate_lattice(
            initial,
            args.neighborhood,
            args.max_rounds,
            rng,
            args.frame_every,
            args.metric_every,
        )
        final_metrics = lattice_metrics(result.rounds, result.final, args.neighborhood)
        print(
            f"sim {sim}: rounds={result.rounds} frozen={result.frozen} "
            f"active_density={final_metrics['active_density']:.3f} "
            f"clusters={final_metrics['cluster_count']} "
            f"largest_cluster_fraction={final_metrics['largest_cluster_fraction']:.3f}"
        )

    if result is None:
        return
    if args.save_metrics is not None:
        save_metrics(args.save_metrics, result.metrics)
    if args.save_final_summary is not None:
        save_final_summary(args.save_final_summary, result, args.neighborhood)
    if args.save_heatmaps is not None:
        save_heatmaps(args.save_heatmaps, result, f"{args.neighborhood}, {args.initial_mode}")
    if args.save_metrics_plot is not None:
        save_metrics_plot(args.save_metrics_plot, result.metrics)
    if args.save_animation is not None:
        save_animation(args.save_animation, result.frames, args.interval_ms)


if __name__ == "__main__":
    main()
