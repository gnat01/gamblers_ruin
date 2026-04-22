# Summary

## Model

We implemented a multi-gambler ruin simulator.

- `N` gamblers, currently assumed even.
- Gambler `i` starts with wealth `A_i`.
- Each round pairs active gamblers.
- Each pair plays a fair dollar-transfer game: winner `+1`, loser `-1`.
- Gamblers at zero are inactive.
- Absorption occurs when only one gambler remains.

For fair play, the eventual winner probabilities are:

```text
P(i wins) = A_i / sum_j A_j
```

So the initially richest gambler wins most often, but not always.

## CLI And Outputs

Main script:

```text
src/gamblers_ruin.py
```

Added CLI support for:

- explicit initial amounts via `--amounts`
- generated gamblers via `--gamblers`, `--total-wealth`, `--amount-mode`
- repeated simulations via `--sims` / `--trials`
- custom seed via `--seed`
- optional finite cap via `--max-rounds`
- CSV survivor table via `--save-table`
- survivor frequency plot via `--save-plot`
- trajectory animation via `--animate --output`

Generated amounts default to descending wealth, so gambler `0` is richest, gambler `1` second richest, etc. This is WLOG for generated runs because labels are arbitrary.

## Pairing Strategies

Implemented:

```text
random
ranked-after-warmup
```

For `ranked-after-warmup`, the first `W` rounds are random. After that, active gamblers are sorted by current wealth and paired:

```text
rank 1 vs rank 2
rank 3 vs rank 4
...
```

Controlled by:

```text
--pairing ranked-after-warmup
--warmup-rounds 3
```

## Trajectory Shape And Hurst

We added optional trajectory and Hurst analysis.

Outputs:

- `--save-trajectory-plot`
- `--save-hurst-table`
- `--save-hurst-plot`

Hurst is computed window-by-window on sample wealth trajectories as a rough quantitative measure of trajectory persistence/roughness. It is intended for comparing strategies and windows, not as an asymptotic theorem.

## Absorption Time Theory

We wrote notes on absorption time:

```text
absorption_time_notes.md
docs/absorption_time_notes.md
```

Main hypothesis:

```text
E[T] ~= C(strategy, N) * sum_{i < j} A_i A_j
```

Equivalently, with total wealth `S` and:

```text
HHI = sum_i (A_i / S)^2
```

we have:

```text
sum_{i < j} A_i A_j = S^2 / 2 * (1 - HHI)
```

So absorption time should increase with initial wealth dispersion and decrease with initial concentration.

## Censoring

If a run uses finite `--max-rounds` and does not absorb, the observation is right-censored:

```text
T > T_max
```

The code records:

```text
observed_time = min(T, T_max)
absorbed = True / False
```

For small systems, we plan to run to true absorption. For larger systems, survival-analysis methods are the right lens.

## Step 1: Small-System Absorption Sweep

Implemented the first empirical step:

```text
--small-sweep-output
--small-sweep-summary
--small-sweep-plot
--sweep-vectors-per-family
--sweep-sims-per-vector
```

The sweep generates multiple initial wealth distributions:

- equal
- linear descending
- geometric
- one whale
- two whales
- Dirichlet-style random splits

It saves one row per simulation with:

```text
N, S, initial amounts, family,
HHI, 1-HHI, pairwise sum, Gini, max share, entropy,
pairing strategy, observed time, absorbed flag, winner
```

It also saves one summary row per initial vector with:

```text
mean / median / p90 / p99 observed time
absorbed count
censored count
censored fraction
absorbed-only absorption-time summaries
```

## Docs

Main run instructions:

```text
how_to_run.md
```

Empirical sweep plan:

```text
docs/empirical_sweep_plan.md
```

The current next step is to run the small-system sweep, inspect the `T` vs `1 - HHI` and `T` vs `sum_{i<j} A_i A_j` plots, then decide whether to scale up or move to strategy comparison.
