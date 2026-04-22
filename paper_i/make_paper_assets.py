from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"


def read_summary(path: Path, strategy: str) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row: dict[str, float | str] = {"strategy": strategy}
            for key, value in raw.items():
                if key in {"family", "amounts", "pairing"}:
                    row[key] = value
                else:
                    row[key] = float(value)
            rows.append(row)
    return rows


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return float(slope), float(intercept), r2


def grouped_family(name: str) -> str:
    if name == "equal":
        return "equal"
    if name.startswith("linear"):
        return "linear"
    if name.startswith("geometric"):
        return "geometric"
    if name.startswith("one_whale"):
        return "one whale"
    if name.startswith("two_whales"):
        return "two whales"
    if name.startswith("dirichlet"):
        return "Dirichlet"
    return name


def add_family_groups(rows: list[dict[str, float | str]]) -> None:
    for row in rows:
        row["family_group"] = grouped_family(str(row["family"]))


def save_concentration_plot(rows: list[dict[str, float | str]]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    strategies = ["random", "ranked-after-warmup"]
    colors = {"random": "#1f77b4", "ranked-after-warmup": "#d62728"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True)
    for strategy in strategies:
        subset = [row for row in rows if row["strategy"] == strategy]
        x = np.array([float(row["one_minus_hhi"]) for row in subset])
        y_median = np.array([float(row["median_absorption_time_absorbed_only"]) / float(row["S"]) ** 2 for row in subset])
        y_mean = np.array([float(row["mean_absorption_time_absorbed_only"]) / float(row["S"]) ** 2 for row in subset])

        for ax, y, label in [(axes[0], y_median, "median"), (axes[1], y_mean, "mean")]:
            ax.scatter(x, y, s=42, alpha=0.8, color=colors[strategy], edgecolor="white", linewidth=0.5, label=strategy)
            slope, intercept, r2 = fit_line(x, y)
            line_x = np.linspace(x.min(), x.max(), 100)
            ax.plot(line_x, slope * line_x + intercept, color=colors[strategy], linewidth=2, alpha=0.9)
            ax.text(
                0.03,
                0.95 - 0.1 * strategies.index(strategy),
                f"{strategy}: $R^2={r2:.2f}$",
                transform=ax.transAxes,
                color=colors[strategy],
                fontsize=9,
                va="top",
            )

    axes[0].set_title("Typical absorption time")
    axes[0].set_ylabel(r"absorption time / $S^2$")
    axes[1].set_title("Mean absorption time")
    for ax in axes:
        ax.set_xlabel(r"initial dispersion, $1-\mathrm{HHI}$")
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "concentration_vs_time.png", dpi=180)
    plt.close(fig)


def save_pairwise_plot(rows: list[dict[str, float | str]]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    colors = {"random": "#1f77b4", "ranked-after-warmup": "#d62728"}
    for strategy in ["random", "ranked-after-warmup"]:
        subset = [row for row in rows if row["strategy"] == strategy]
        x = np.array([float(row["pairwise_sum"]) for row in subset])
        y = np.array([float(row["median_absorption_time_absorbed_only"]) for row in subset])
        ax.scatter(x, y, s=44, alpha=0.8, color=colors[strategy], edgecolor="white", linewidth=0.5, label=strategy)
        slope, intercept, _ = fit_line(x, y)
        line_x = np.linspace(x.min(), x.max(), 100)
        ax.plot(line_x, slope * line_x + intercept, color=colors[strategy], linewidth=2)
    ax.set_xlabel(r"$\sum_{i<j} A_i A_j$")
    ax.set_ylabel("median absorption time")
    ax.set_title("Pairwise wealth-product predictor")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "pairwise_product_vs_time.png", dpi=180)
    plt.close(fig)


def save_family_plot(rows: list[dict[str, float | str]]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    family_order = ["equal", "linear", "geometric", "one whale", "two whales", "Dirichlet"]
    strategy_order = ["random", "ranked-after-warmup"]
    width = 0.36
    x = np.arange(len(family_order))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for offset, strategy in [(-width / 2, "random"), (width / 2, "ranked-after-warmup")]:
        means = []
        errors = []
        for family in family_order:
            values = [
                float(row["median_absorption_time_absorbed_only"]) / float(row["S"]) ** 2
                for row in rows
                if row["strategy"] == strategy and row["family_group"] == family
            ]
            means.append(float(np.mean(values)) if values else float("nan"))
            errors.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
        ax.bar(x + offset, means, width=width, yerr=errors, capsize=3, label=strategy)
    ax.set_xticks(x)
    ax.set_xticklabels(family_order, rotation=20, ha="right")
    ax.set_ylabel(r"median absorption time / $S^2$")
    ax.set_title("Absorption time by distribution family")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "family_absorption_bars.png", dpi=180)
    plt.close(fig)


def save_regression_table(rows: list[dict[str, float | str]]) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Strategy & Vectors & Slope & Intercept & $R^2$ \\",
        r"\midrule",
    ]
    for strategy in ["random", "ranked-after-warmup"]:
        subset = [row for row in rows if row["strategy"] == strategy]
        x = np.array([float(row["one_minus_hhi"]) for row in subset])
        y = np.array([float(row["median_absorption_time_absorbed_only"]) / float(row["S"]) ** 2 for row in subset])
        slope, intercept, r2 = fit_line(x, y)
        label = "Ranked after warmup" if strategy == "ranked-after-warmup" else "Random"
        lines.append(f"{label} & {len(subset)} & {slope:.3f} & {intercept:.3f} & {r2:.3f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "regression_table.tex").write_text("\n".join(lines) + "\n")


def save_family_table(rows: list[dict[str, float | str]]) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    family_order = ["equal", "linear", "geometric", "one whale", "two whales", "Dirichlet"]
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Family & Vectors & Mean $1-\mathrm{HHI}$ & Random med. $T/S^2$ & Ranked med. $T/S^2$ \\",
        r"\midrule",
    ]
    for family in family_order:
        family_rows = [row for row in rows if row["family_group"] == family]
        random_values = [
            float(row["median_absorption_time_absorbed_only"]) / float(row["S"]) ** 2
            for row in family_rows
            if row["strategy"] == "random"
        ]
        ranked_values = [
            float(row["median_absorption_time_absorbed_only"]) / float(row["S"]) ** 2
            for row in family_rows
            if row["strategy"] == "ranked-after-warmup"
        ]
        dispersion = np.mean([float(row["one_minus_hhi"]) for row in family_rows])
        lines.append(
            f"{family.title()} & {len(family_rows) // 2} & {dispersion:.3f} & "
            f"{np.mean(random_values):.3f} & {np.mean(ranked_values):.3f} " + r"\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "family_table.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    random_rows = read_summary(DATA / "random_summary.csv", "random")
    ranked_rows = read_summary(DATA / "ranked_summary.csv", "ranked-after-warmup")
    rows = random_rows + ranked_rows
    add_family_groups(rows)
    save_concentration_plot(rows)
    save_pairwise_plot(rows)
    save_family_plot(rows)
    save_regression_table(rows)
    save_family_table(rows)


if __name__ == "__main__":
    main()
