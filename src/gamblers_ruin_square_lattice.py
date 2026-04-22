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
    local_matches: int = 0
    nonlocal_matches: int = 0


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


def add_nonlocal_edges(
    local_edges: np.ndarray,
    side: int,
    probability: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if probability < 0 or probability > 1:
        raise ValueError("nonlocal probability must be between 0 and 1.")
    site_count = side * side
    edge_set = {tuple(edge) for edge in local_edges.tolist()}
    nonlocal_edges = []
    local_neighbors = [set() for _ in range(site_count)]
    for a, b in edge_set:
        local_neighbors[a].add(b)
        local_neighbors[b].add(a)

    all_sites = np.arange(site_count)
    for v in range(site_count):
        if rng.random() > probability:
            continue
        forbidden = local_neighbors[v] | {v}
        candidates = np.array([u for u in all_sites if int(u) not in forbidden], dtype=int)
        if len(candidates) == 0:
            continue
        u = int(rng.choice(candidates))
        edge = (min(v, u), max(v, u))
        if edge not in edge_set:
            edge_set.add(edge)
            nonlocal_edges.append(edge)

    if nonlocal_edges:
        all_edges = np.vstack([local_edges, np.array(nonlocal_edges, dtype=int)])
        is_nonlocal = np.concatenate([np.zeros(len(local_edges), dtype=bool), np.ones(len(nonlocal_edges), dtype=bool)])
    else:
        all_edges = local_edges.copy()
        is_nonlocal = np.zeros(len(local_edges), dtype=bool)
    return all_edges, is_nonlocal


def active_edges(amounts: np.ndarray, edges: np.ndarray) -> np.ndarray:
    flat = amounts.ravel()
    mask = (flat[edges[:, 0]] > 0) & (flat[edges[:, 1]] > 0)
    return edges[mask]


def active_edges_with_types(amounts: np.ndarray, edges: np.ndarray, is_nonlocal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = amounts.ravel()
    mask = (flat[edges[:, 0]] > 0) & (flat[edges[:, 1]] > 0)
    return edges[mask], is_nonlocal[mask]


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


def random_matching_with_types(
    active_edge_array: np.ndarray,
    active_is_nonlocal: np.ndarray,
    site_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if len(active_edge_array) == 0:
        return active_edge_array, active_is_nonlocal
    order = rng.permutation(len(active_edge_array))
    used = np.zeros(site_count, dtype=bool)
    matched = []
    matched_types = []
    for idx in order:
        a, b = active_edge_array[idx]
        if not used[a] and not used[b]:
            used[a] = True
            used[b] = True
            matched.append((a, b))
            matched_types.append(bool(active_is_nonlocal[idx]))
    if not matched:
        return np.empty((0, 2), dtype=int), np.empty(0, dtype=bool)
    return np.array(matched, dtype=int), np.array(matched_types, dtype=bool)


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


def wealth_concentration(amounts: np.ndarray) -> dict[str, float]:
    flat = amounts.ravel().astype(float)
    total = float(flat.sum())
    if total <= 0:
        return {"hhi": float("nan"), "one_minus_hhi": float("nan"), "gini": float("nan"), "max_share": float("nan")}
    shares = flat / total
    sorted_values = np.sort(flat)
    n = len(sorted_values)
    gini = float((2 * np.sum(np.arange(1, n + 1) * sorted_values)) / (n * total) - (n + 1) / n)
    hhi = float(np.sum(shares**2))
    return {
        "hhi": hhi,
        "one_minus_hhi": 1 - hhi,
        "gini": gini,
        "max_share": float(shares.max()),
    }


def hhi_of_weights(weights: np.ndarray) -> float:
    shares = weights / weights.sum()
    return float(np.sum(shares**2))


def initial_lattice_with_target_hhi(
    side: int,
    total_wealth: int,
    target_hhi: float,
    rng: np.random.Generator,
    tolerance: float,
    max_attempts: int,
) -> tuple[np.ndarray, float]:
    sites = side * side
    min_hhi = 1 / sites
    max_hhi = ((total_wealth - sites + 1) / total_wealth) ** 2 + (sites - 1) * (1 / total_wealth) ** 2
    if target_hhi < min_hhi - tolerance or target_hhi > max_hhi + tolerance:
        raise ValueError(f"target HHI {target_hhi:.4f} is outside feasible range [{min_hhi:.4f}, {max_hhi:.4f}]")

    best_amounts: np.ndarray | None = None
    best_error = float("inf")
    best_hhi = float("nan")
    uniform = np.ones(sites)

    for _ in range(max_attempts):
        spike = rng.dirichlet(np.full(sites, 0.08))
        if hhi_of_weights(spike) < target_hhi:
            hotspot = rng.integers(sites)
            spike = np.full(sites, 1e-9)
            spike[hotspot] = 1.0

        low, high = 0.0, 1.0
        for _ in range(40):
            mix = (low + high) / 2
            weights = (1 - mix) * uniform + mix * spike
            hhi = hhi_of_weights(weights)
            if hhi < target_hhi:
                low = mix
            else:
                high = mix

        weights = (1 - high) * uniform + high * spike
        amounts = np.floor((total_wealth - sites) * weights / weights.sum()).astype(int) + 1
        remainder = total_wealth - int(amounts.sum())
        if remainder:
            raw = (total_wealth - sites) * weights / weights.sum()
            order = np.argsort(raw - np.floor(raw))[::-1]
            amounts[order[:remainder]] += 1

        measured_hhi = wealth_concentration(amounts.reshape(side, side))["hhi"]
        error = abs(measured_hhi - target_hhi)
        if error < best_error:
            best_amounts = amounts.copy()
            best_error = error
            best_hhi = measured_hhi
        if error <= tolerance:
            break

    if best_amounts is None:
        raise RuntimeError("failed to generate target-HHI lattice")
    return best_amounts.reshape(side, side), best_hhi


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


def simulate_lattice_on_edges(
    initial: np.ndarray,
    edges: np.ndarray,
    is_nonlocal: np.ndarray,
    neighborhood: str,
    max_rounds: int,
    rng: np.random.Generator,
    metric_every: int,
) -> LatticeResult:
    amounts = initial.copy()
    side = amounts.shape[0]
    metrics = [lattice_metrics(0, amounts, neighborhood)]
    frozen = False
    rounds = 0
    local_matches = 0
    nonlocal_matches = 0

    while max_rounds <= 0 or rounds < max_rounds:
        active_edge_array, active_types = active_edges_with_types(amounts, edges, is_nonlocal)
        if len(active_edge_array) == 0:
            frozen = True
            break

        matching, matched_types = random_matching_with_types(active_edge_array, active_types, side * side, rng)
        if len(matching) == 0:
            frozen = True
            break

        flat = amounts.ravel()
        first_wins = rng.integers(2, size=len(matching), dtype=bool)
        winners = np.where(first_wins, matching[:, 0], matching[:, 1])
        losers = np.where(first_wins, matching[:, 1], matching[:, 0])
        np.add.at(flat, winners, 1)
        np.add.at(flat, losers, -1)
        nonlocal_matches += int(matched_types.sum())
        local_matches += int(len(matched_types) - matched_types.sum())
        rounds += 1

        if metric_every > 0 and rounds % metric_every == 0:
            metrics.append(lattice_metrics(rounds, amounts, neighborhood))

        if np.count_nonzero(amounts) <= 1:
            frozen = False
            break

    if int(metrics[-1]["round"]) != rounds:
        metrics.append(lattice_metrics(rounds, amounts, neighborhood))
    return LatticeResult(
        initial=initial,
        final=amounts,
        frames=[initial.copy(), amounts.copy()],
        metrics=metrics,
        frozen=frozen,
        rounds=rounds,
        local_matches=local_matches,
        nonlocal_matches=nonlocal_matches,
    )


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


def save_wealth_histogram(path: Path, result: LatticeResult, bins: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    initial_all = result.initial.ravel()
    final_all = result.final.ravel()
    final_active = final_all[final_all > 0]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].hist(initial_all, bins=bins, alpha=0.65, label="initial", color="#1f77b4")
    axes[0].hist(final_all, bins=bins, alpha=0.65, label="final incl. zeros", color="#d62728")
    axes[0].set_title("All sites")
    axes[0].set_xlabel("wealth")
    axes[0].set_ylabel("site count")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    if len(final_active):
        axes[1].hist(final_active, bins=bins, color="#2ca02c")
    axes[1].set_title("Final active sites only")
    axes[1].set_xlabel("wealth")
    axes[1].set_ylabel("site count")
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f"saved wealth histogram to {path}")


def cluster_count_by_size(sizes: list[int]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for size in sizes:
        counts[size] = counts.get(size, 0) + 1
    return counts


def aggregate_cluster_counts(cluster_sizes_by_sim: list[list[int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_sizes = sorted({size for sizes in cluster_sizes_by_sim for size in sizes})
    if not all_sizes:
        return np.array([]), np.array([]), np.array([]), np.array([])
    matrix = np.zeros((len(cluster_sizes_by_sim), len(all_sizes)), dtype=float)
    for sim, sizes in enumerate(cluster_sizes_by_sim):
        counts = cluster_count_by_size(sizes)
        for idx, size in enumerate(all_sizes):
            matrix[sim, idx] = counts.get(size, 0)
    return (
        np.array(all_sizes, dtype=int),
        matrix.mean(axis=0),
        np.percentile(matrix, 5, axis=0),
        np.percentile(matrix, 95, axis=0),
    )


def loglog_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    mask = (x > 0) & (y > 0)
    log_x = np.log(x[mask])
    log_y = np.log(y[mask])
    if len(log_x) < 2:
        return float("nan"), float("nan"), float("nan"), np.full_like(x, np.nan, dtype=float)
    slope, intercept = np.polyfit(log_x, log_y, 1)
    fitted_positive = np.exp(intercept + slope * log_x)
    ss_res = float(np.sum((log_y - (intercept + slope * log_x)) ** 2))
    ss_tot = float(np.sum((log_y - log_y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    fitted = np.full_like(x, np.nan, dtype=float)
    fitted[mask] = fitted_positive
    return float(slope), float(intercept), float(r2), fitted


def save_cluster_size_table(path: Path, cluster_sizes_by_sim: list[list[int]]) -> None:
    sizes, means, lows, highs = aggregate_cluster_counts(cluster_sizes_by_sim)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cluster_size", "mean_cluster_count", "p05_cluster_count", "p95_cluster_count"])
        for size, mean, low, high in zip(sizes, means, lows, highs):
            writer.writerow([int(size), mean, low, high])
    print(f"saved cluster size table to {path}")


def save_cluster_size_plot(path: Path, result: LatticeResult, cluster_sizes_by_sim: list[list[int]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    final_sizes = np.array(cluster_sizes_by_sim[-1], dtype=int) if cluster_sizes_by_sim else np.array([], dtype=int)
    sizes, means, lows, highs = aggregate_cluster_counts(cluster_sizes_by_sim)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    mask_ax = axes[0, 0]
    hist_ax = axes[0, 1]
    loglog_ax = axes[0, 2]
    mask_ax.imshow(result.final > 0, cmap="gray_r", interpolation="nearest")
    mask_ax.set_title("final active mask")
    mask_ax.set_xticks([])
    mask_ax.set_yticks([])

    if len(final_sizes):
        unique_sizes = np.unique(final_sizes)
        hist_ax.hist(final_sizes, bins=min(40, max(5, len(unique_sizes))), color="#4c78a8")
        hist_ax.set_xlabel("cluster size")
        hist_ax.set_ylabel("cluster count")
        hist_ax.set_title("last simulation histogram")
        hist_ax.grid(axis="y", alpha=0.25)

        positive = means > 0
        loglog_ax.fill_between(sizes[positive], lows[positive], highs[positive], color="black", alpha=0.18, label="90% band")
        loglog_ax.loglog(sizes[positive], means[positive], "o-", color="red", linewidth=2, label="mean over sims")
        left_mask = positive & (sizes >= 1) & (sizes <= 10)
        right_mask = positive & (sizes > 10)
        left_fit = loglog_fit(sizes[left_mask], means[left_mask]) if np.count_nonzero(left_mask) >= 2 else None
        right_fit = loglog_fit(sizes[right_mask], means[right_mask]) if np.count_nonzero(right_mask) >= 2 else None

        annotations = []
        if left_fit is not None:
            left_slope, _, left_r2, left_fitted = left_fit
            loglog_ax.loglog(
                sizes[left_mask],
                left_fitted,
                "--",
                color="blue",
                linewidth=2,
                label="fit: 1-10",
            )
            annotations.append(rf"1-10: slope={left_slope:.2f}, $R^2={left_r2:.2f}$")
        if right_fit is not None:
            right_slope, _, right_r2, right_fitted = right_fit
            loglog_ax.loglog(
                sizes[right_mask],
                right_fitted,
                "--",
                color="purple",
                linewidth=2,
                label="fit: >10",
            )
            annotations.append(rf">10: slope={right_slope:.2f}, $R^2={right_r2:.2f}$")
        loglog_ax.axvline(10, color="gray", linestyle=":", linewidth=1.5)

        if left_fit is None and right_fit is None:
            slope, _, r2, fitted = loglog_fit(sizes, means)
            if not np.isnan(r2):
                loglog_ax.loglog(
                    sizes[positive],
                    fitted[positive],
                    "--",
                    color="blue",
                    linewidth=2,
                    label="global fit",
                )
                annotations.append(rf"global: slope={slope:.2f}, $R^2={r2:.2f}$")
        loglog_ax.set_xlabel("cluster size")
        loglog_ax.set_ylabel("number of clusters")
        loglog_ax.set_title("aggregate log-log distribution")
        loglog_ax.grid(True, which="both", alpha=0.25)
        loglog_ax.legend(frameon=False, loc="upper right", fontsize=8)
        if annotations:
            loglog_ax.text(
                0.5,
                0.96,
                "\n".join(annotations),
                transform=loglog_ax.transAxes,
                fontsize=7,
                va="top",
                ha="center",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
            )
    else:
        for ax in (hist_ax, loglog_ax):
            ax.text(0.5, 0.5, "no active clusters", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])

    for ax in axes[1, :]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f"saved cluster size plot to {path}")


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


def binned_curve(
    rows: list[dict[str, float | int | bool]],
    key: str,
    bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.array([float(row["initial_hhi"]) for row in rows])
    y = np.array([float(row[key]) for row in rows])
    edges = np.linspace(float(x.min()), float(x.max()), bins + 1)
    centers = []
    means = []
    lows = []
    highs = []
    for i in range(bins):
        if i == bins - 1:
            mask = (x >= edges[i]) & (x <= edges[i + 1])
        else:
            mask = (x >= edges[i]) & (x < edges[i + 1])
        if not np.any(mask):
            continue
        centers.append(float(np.mean(x[mask])))
        means.append(float(np.mean(y[mask])))
        lows.append(float(np.percentile(y[mask], 5)))
        highs.append(float(np.percentile(y[mask], 95)))
    return np.array(centers), np.array(means), np.array(lows), np.array(highs)


def run_bifurcation(args: argparse.Namespace, seed: int | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    total_wealth = args.total_wealth if args.total_wealth is not None else args.N * args.N * args.initial_wealth
    values = np.arange(args.bifurcation_hhi_min, args.bifurcation_hhi_max + args.bifurcation_hhi_step / 2, args.bifurcation_hhi_step)
    rows = []
    for target_hhi in values:
        for sim in range(args.sims):
            initial, measured_target_hhi = initial_lattice_with_target_hhi(
                args.N,
                total_wealth,
                float(target_hhi),
                rng,
                args.bifurcation_hhi_tolerance,
                args.bifurcation_hhi_attempts,
            )
            initial_concentration = wealth_concentration(initial)
            result = simulate_lattice(
                initial,
                args.neighborhood,
                args.max_rounds,
                rng,
                frame_every=0,
                metric_every=max(1, args.metric_every),
            )
            final_metrics = lattice_metrics(result.rounds, result.final, args.neighborhood)
            final_concentration = wealth_concentration(result.final)
            rows.append(
                {
                    "target_hhi": float(target_hhi),
                    "measured_target_hhi": measured_target_hhi,
                    "sim": sim,
                    "initial_hhi": initial_concentration["hhi"],
                    "initial_one_minus_hhi": initial_concentration["one_minus_hhi"],
                    "initial_gini": initial_concentration["gini"],
                    "initial_max_share": initial_concentration["max_share"],
                    "final_hhi": final_concentration["hhi"],
                    "final_one_minus_hhi": final_concentration["one_minus_hhi"],
                    "final_gini": final_concentration["gini"],
                    "final_max_share": final_concentration["max_share"],
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
            ax.scatter([row["initial_hhi"] for row in rows], [row[key] for row in rows], s=14, alpha=0.35, color="#4c78a8")
            centers, means, lows, highs = binned_curve(rows, key, args.bifurcation_bins)
            if len(centers):
                ax.fill_between(centers, lows, highs, color="black", alpha=0.18, label="90% band")
                ax.plot(centers, means, color="red", linewidth=2.4, label="bin mean")
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.25)
        axes[0, 0].legend(frameon=False, loc="best")
        for ax in axes[-1]:
            ax.set_xlabel("initial HHI")
        fig.suptitle(f"Lattice ruin bifurcation scan by initial HHI ({args.neighborhood})")
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


def fit_exponential_decay(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    mask = (y > 0) & np.isfinite(y)
    if np.count_nonzero(mask) < 2:
        return float("nan"), float("nan"), float("nan"), np.full_like(y, np.nan, dtype=float)
    log_y = np.log(y[mask])
    slope, intercept = np.polyfit(x[mask], log_y, 1)
    pred_log = slope * x[mask] + intercept
    ss_res = float(np.sum((log_y - pred_log) ** 2))
    ss_tot = float(np.sum((log_y - log_y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    fitted = np.full_like(y, np.nan, dtype=float)
    fitted[mask] = np.exp(pred_log)
    return float(slope), float(intercept), float(r2), fitted


def summarize_by_p(p: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unique_p = np.array(sorted(set(float(value) for value in p)))
    means = []
    lows = []
    highs = []
    for value in unique_p:
        mask = np.isclose(p, value)
        group = values[mask]
        means.append(float(np.mean(group)))
        lows.append(float(np.percentile(group, 5)))
        highs.append(float(np.percentile(group, 95)))
    return unique_p, np.array(means), np.array(lows), np.array(highs)


def save_nonlocal_scan_plot(path: Path, rows: list[dict[str, float | int | bool]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    p = np.array([float(row["p"]) for row in rows])
    times = np.array([float(row["rounds"]) for row in rows])
    active_density = np.array([float(row["final_active_density"]) for row in rows])
    cluster_count = np.array([float(row["final_cluster_count"]) for row in rows])
    largest_fraction = np.array([float(row["final_largest_cluster_fraction"]) for row in rows])
    realized_nonlocal = np.array([float(row["realized_nonlocal_match_fraction"]) for row in rows])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2))
    specs = [
        (times, "rounds to absorb/freeze/cap", True),
        (active_density, "final active density", False),
        (cluster_count, "final cluster count", False),
        (largest_fraction, "largest cluster / active", False),
        (realized_nonlocal, "realized nonlocal match fraction", False),
    ]
    for ax, (values, label, log_y) in zip(axes.ravel()[:5], specs):
        unique_p, means, lows, highs = summarize_by_p(p, values)
        ax.scatter(p, values, color="#4c78a8", s=14, alpha=0.28, label="runs")
        ax.fill_between(unique_p, lows, highs, color="black", alpha=0.16, label="90% band")
        ax.plot(unique_p, means, "o-", color="red", linewidth=2.2, markersize=4, label="mean")
        if log_y:
            ax.set_yscale("log")
            slope, _, r2, fitted = fit_exponential_decay(unique_p, means)
            if not np.isnan(r2):
                ax.plot(unique_p, fitted, "--", color="blue", linewidth=2, label=rf"log-mean fit $R^2={r2:.2f}$")
        ax.set_xlabel("nonlocal edge probability p")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, loc="best", fontsize=8)

    axes.ravel()[5].axis("off")
    fig.suptitle("Small-world nonlocality scan")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f"saved nonlocal scan plot to {path}")


def save_nonlocal_trajectory_plot(path: Path, selected: dict[float, list[dict[str, float | int]]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), squeeze=False)
    keys = [
        ("active_density", "active density"),
        ("cluster_count", "cluster count"),
        ("largest_cluster_fraction", "largest cluster / active"),
    ]
    for col, (key, label) in enumerate(keys):
        ax = axes[0, col]
        for p, metrics in selected.items():
            rounds = [row["round"] for row in metrics]
            values = [row[key] for row in metrics]
            ax.plot(rounds, values, linewidth=1.8, label=f"p={p:.2f}")
        ax.set_xlabel("round")
        ax.set_ylabel(label)
        ax.set_title(f"{label} over time")
        ax.grid(True, alpha=0.25)

        log_ax = axes[1, col]
        for p, metrics in selected.items():
            rounds = np.array([row["round"] for row in metrics], dtype=float)
            values = np.array([row[key] for row in metrics], dtype=float)
            mask = (rounds > 0) & (values > 0)
            if np.any(mask):
                log_ax.loglog(rounds[mask], values[mask], linewidth=1.8, label=f"p={p:.2f}")
        log_ax.set_xlabel("round")
        log_ax.set_ylabel(label)
        log_ax.set_title(f"log-log {label}")
        log_ax.grid(True, which="both", alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Pattern metrics for selected nonlocal probabilities")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f"saved nonlocal trajectory plot to {path}")


def run_nonlocal_scan(args: argparse.Namespace, seed: int | None) -> None:
    rng = np.random.default_rng(seed)
    initial_lattices = [
        initial_lattice(
            args.N,
            args.initial_mode,
            args.initial_wealth,
            args.total_wealth,
            args.heterogeneity,
            args.seed_count,
            rng,
        )
        for _ in range(args.sims)
    ]
    local_edges = neighbor_edges(args.N, args.neighborhood)
    p_values = np.arange(args.nonlocal_p_min, args.nonlocal_p_max + args.nonlocal_p_step / 2, args.nonlocal_p_step)
    if args.nonlocal_selected_p is None:
        selected_p = np.linspace(args.nonlocal_p_min, args.nonlocal_p_max, args.nonlocal_selected_count).tolist()
    else:
        selected_p = [float(value) for value in args.nonlocal_selected_p]
    rows = []
    selected_metrics: dict[float, list[dict[str, float | int]]] = {}

    for p in p_values:
        for sim, initial in enumerate(initial_lattices):
            if seed is None:
                graph_rng = np.random.default_rng()
                sim_rng = np.random.default_rng()
            else:
                p_seed = int(round(float(p) * 100_000))
                graph_rng = np.random.default_rng(seed + p_seed + sim * 1_000_003 + 17)
                sim_rng = np.random.default_rng(seed + p_seed + sim * 1_000_003 + 997)
            edges, is_nonlocal = add_nonlocal_edges(local_edges, args.N, float(p), graph_rng)
            result = simulate_lattice_on_edges(
                initial,
                edges,
                is_nonlocal,
                args.neighborhood,
                args.max_rounds,
                sim_rng,
                args.metric_every,
            )
            final_metrics = lattice_metrics(result.rounds, result.final, args.neighborhood)
            active_sites = int(final_metrics["active_sites"])
            absorbed = active_sites <= 1
            capped = args.max_rounds > 0 and result.rounds >= args.max_rounds and not absorbed and not result.frozen
            total_matches = result.local_matches + result.nonlocal_matches
            row = {
                "p": float(p),
                "sim": sim,
                "local_edges": int(len(local_edges)),
                "nonlocal_edges": int(is_nonlocal.sum()),
                "mean_degree": float(2 * len(edges) / (args.N * args.N)),
                "rounds": result.rounds,
                "absorbed": absorbed,
                "frozen": result.frozen,
                "capped": capped,
                "local_matches": result.local_matches,
                "nonlocal_matches": result.nonlocal_matches,
                "realized_nonlocal_match_fraction": result.nonlocal_matches / total_matches if total_matches else 0.0,
                "final_active_density": final_metrics["active_density"],
                "final_cluster_count": final_metrics["cluster_count"],
                "final_largest_cluster_fraction": final_metrics["largest_cluster_fraction"],
                "final_interface_length": final_metrics["interface_length"],
                "winner_site": int(np.argmax(result.final)) if absorbed else -1,
            }
            rows.append(row)
            if sim == 0 and any(abs(float(p) - value) <= args.nonlocal_p_step / 2 for value in selected_p):
                selected_metrics[float(p)] = result.metrics
            print(
                f"p={p:.2f} sim={sim}: rounds={result.rounds} absorbed={absorbed} frozen={result.frozen} "
                f"nonlocal_edges={int(is_nonlocal.sum())}"
            )

    if args.nonlocal_table is not None:
        args.nonlocal_table.parent.mkdir(parents=True, exist_ok=True)
        with args.nonlocal_table.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved nonlocal scan table to {args.nonlocal_table}")
    if args.nonlocal_plot is not None:
        save_nonlocal_scan_plot(args.nonlocal_plot, rows)
    if args.nonlocal_trajectory_plot is not None and selected_metrics:
        save_nonlocal_trajectory_plot(args.nonlocal_trajectory_plot, selected_metrics)


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
    parser.add_argument("--save-wealth-histogram", type=Path, help="Save initial/final wealth histograms.")
    parser.add_argument("--save-cluster-size-table", type=Path, help="Save final cluster-size distribution as CSV.")
    parser.add_argument("--save-cluster-size-plot", type=Path, help="Save final cluster-size distribution plots.")
    parser.add_argument("--histogram-bins", type=int, default=40, help="Number of bins for wealth histograms.")
    parser.add_argument("--interval-ms", type=int, default=80, help="Animation frame interval.")
    parser.add_argument("--bifurcation-output", type=Path, help="Save target-HHI bifurcation plot.")
    parser.add_argument("--bifurcation-table", type=Path, help="Save target-HHI bifurcation raw CSV.")
    parser.add_argument("--bifurcation-hhi-min", type=float, default=0.01)
    parser.add_argument("--bifurcation-hhi-max", type=float, default=0.50)
    parser.add_argument("--bifurcation-hhi-step", type=float, default=0.01)
    parser.add_argument("--bifurcation-hhi-tolerance", type=float, default=0.002)
    parser.add_argument("--bifurcation-hhi-attempts", type=int, default=30)
    parser.add_argument("--bifurcation-bins", type=int, default=8, help="Representative HHI bins for red mean curve and 90%% band.")
    parser.add_argument("--nonlocal-scan", action="store_true", help="Run static small-world nonlocal edge scan.")
    parser.add_argument("--nonlocal-p-min", type=float, default=0.01)
    parser.add_argument("--nonlocal-p-max", type=float, default=0.40)
    parser.add_argument("--nonlocal-p-step", type=float, default=0.01)
    parser.add_argument("--nonlocal-table", type=Path, help="Save nonlocal scan table.")
    parser.add_argument("--nonlocal-plot", type=Path, help="Save nonlocal scan summary plot.")
    parser.add_argument("--nonlocal-trajectory-plot", type=Path, help="Save selected-p trajectory comparison plot.")
    parser.add_argument(
        "--nonlocal-selected-p",
        type=float,
        nargs="*",
        default=None,
        help="Selected p values for trajectory plots.",
    )
    parser.add_argument("--nonlocal-selected-count", type=int, default=6, help="Default number of evenly spaced p values for trajectory plots.")
    args = parser.parse_args()

    if args.sims < 1:
        parser.error("--sims must be at least 1.")
    if args.frame_every < 0 or args.metric_every < 1:
        parser.error("--frame-every must be nonnegative and --metric-every must be positive.")
    if args.histogram_bins < 1:
        parser.error("--histogram-bins must be positive.")
    if args.bifurcation_hhi_step <= 0:
        parser.error("--bifurcation-hhi-step must be positive.")
    if args.bifurcation_hhi_min >= args.bifurcation_hhi_max:
        parser.error("--bifurcation-hhi-min must be less than --bifurcation-hhi-max.")
    if args.bifurcation_hhi_attempts < 1:
        parser.error("--bifurcation-hhi-attempts must be at least 1.")
    if args.bifurcation_bins < 2:
        parser.error("--bifurcation-bins must be at least 2.")
    if args.nonlocal_p_step <= 0:
        parser.error("--nonlocal-p-step must be positive.")
    if args.nonlocal_p_min < 0 or args.nonlocal_p_max > 1 or args.nonlocal_p_min > args.nonlocal_p_max:
        parser.error("nonlocal p range must satisfy 0 <= min <= max <= 1.")
    if args.nonlocal_selected_count < 2:
        parser.error("--nonlocal-selected-count must be at least 2.")

    seed = None if args.seed == -1 else args.seed
    if args.bifurcation_output is not None or args.bifurcation_table is not None:
        run_bifurcation(args, seed)
        return
    if args.nonlocal_scan:
        run_nonlocal_scan(args, seed)
        return

    rng = np.random.default_rng(seed)
    result = None
    cluster_sizes_by_sim: list[list[int]] = []
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
        cluster_sizes_by_sim.append(component_sizes(result.final > 0, args.neighborhood))
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
    if args.save_wealth_histogram is not None:
        save_wealth_histogram(args.save_wealth_histogram, result, args.histogram_bins)
    if args.save_cluster_size_table is not None:
        save_cluster_size_table(args.save_cluster_size_table, cluster_sizes_by_sim)
    if args.save_cluster_size_plot is not None:
        save_cluster_size_plot(args.save_cluster_size_plot, result, cluster_sizes_by_sim)
    if args.save_animation is not None:
        save_animation(args.save_animation, result.frames, args.interval_ms)


if __name__ == "__main__":
    main()
