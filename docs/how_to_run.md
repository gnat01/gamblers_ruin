# How To Run

Run commands from the repo root:

```bash
cd /Users/gn/work/learn/python/gamblers_ruin
```

## Install Dependencies

```bash
python -m pip install -r docs/requirements.txt
```

## Run With Explicit Initial Amounts

Use this when you want to specify the exact `A_i` values.

```bash
python src/gamblers_ruin.py --amounts 50,25,15,10 --sims 1000 --seed 7
```

`--trials` is the same as `--sims`:

```bash
python src/gamblers_ruin.py --amounts 50,25,15,10 --trials 1000 --seed 7
```

## Run With Any Even Number Of Gamblers

Use `--gamblers` when you want the script to generate initial amounts.

```bash
python src/gamblers_ruin.py --gamblers 20 --total-wealth 1000 --sims 1000 --seed 7
```

The first model assumes an even number of gamblers. `--gamblers 20` works; `--gamblers 21` intentionally fails.

By default, each simulation runs until true absorption. With large total wealth, this can take a while. To force a cap, use `--max-rounds`; capped trials are reported as `wins/leader-at-cap` instead of silently pretending they fully absorbed.

```bash
python src/gamblers_ruin.py --gamblers 20 --total-wealth 1000 --sims 1000 --max-rounds 1000000
```

## Generated Initial Amount Modes

Descending amounts, richest gambler first:

```bash
python src/gamblers_ruin.py --gamblers 10 --total-wealth 500 --amount-mode descending --sims 1000
```

Equal initial amounts:

```bash
python src/gamblers_ruin.py --gamblers 10 --total-wealth 500 --amount-mode equal --sims 1000
```

Random positive initial amounts:

```bash
python src/gamblers_ruin.py --gamblers 10 --total-wealth 500 --amount-mode random --sims 1000 --seed 7
```

## Pairing Strategies

Plain random pairing:

```bash
python src/gamblers_ruin.py --gamblers 10 --total-wealth 300 --pairing random --sims 500
```

Ranked pairing after a random warmup:

```bash
python src/gamblers_ruin.py --gamblers 10 --total-wealth 300 --pairing ranked-after-warmup --warmup-rounds 3 --sims 500
```

The ranked strategy uses random pairing for the first `--warmup-rounds`, then sorts active gamblers by current wealth and pairs:

```text
rank 1 vs rank 2
rank 3 vs rank 4
rank 5 vs rank 6
...
```

## Save Table And Plot

Save the terminal summary as a CSV table:

```bash
python src/gamblers_ruin.py --gamblers 6 --total-wealth 100 --sims 100 --save-table outputs/table.csv
```

Save an expected-vs-observed survivor frequency plot:

```bash
MPLCONFIGDIR=.mplconfig python src/gamblers_ruin.py --gamblers 6 --total-wealth 100 --sims 100 --save-plot outputs/frequencies.png
```

Save both in one run:

```bash
MPLCONFIGDIR=.mplconfig python src/gamblers_ruin.py --gamblers 6 --total-wealth 100 --sims 100 --save-table outputs/table.csv --save-plot outputs/frequencies.png
```

## Save Trajectory And Hurst Outputs

Save one sample trajectory as a PNG:

```bash
MPLCONFIGDIR=.mplconfig python src/gamblers_ruin.py --gamblers 6 --total-wealth 100 --sims 100 --pairing ranked-after-warmup --warmup-rounds 3 --save-trajectory-plot outputs/trajectory.png
```

Save windowed Hurst estimates as a CSV table:

```bash
python src/gamblers_ruin.py --gamblers 6 --total-wealth 100 --sims 100 --pairing ranked-after-warmup --warmup-rounds 3 --save-hurst-table outputs/hurst.csv
```

Save the windowed Hurst summary plot:

```bash
MPLCONFIGDIR=.mplconfig python src/gamblers_ruin.py --gamblers 6 --total-wealth 100 --sims 100 --pairing ranked-after-warmup --warmup-rounds 3 --save-hurst-plot outputs/hurst.png
```

Control the Hurst windowing:

```bash
MPLCONFIGDIR=.mplconfig python src/gamblers_ruin.py \
  --gamblers 6 \
  --total-wealth 100 \
  --sims 100 \
  --pairing ranked-after-warmup \
  --warmup-rounds 3 \
  --hurst-samples 5 \
  --hurst-window-size 256 \
  --hurst-step-size 128 \
  --save-hurst-table outputs/hurst.csv \
  --save-hurst-plot outputs/hurst.png
```

Small games may absorb before a large Hurst window fits. If the Hurst plot is skipped, lower `--hurst-window-size`.

## Save A Ranked-Strategy Report

This writes the survivor table, survivor frequency plot, sample trajectory plot, Hurst table, and Hurst plot:

```bash
MPLCONFIGDIR=.mplconfig python src/gamblers_ruin.py \
  --gamblers 8 \
  --total-wealth 200 \
  --sims 200 \
  --pairing ranked-after-warmup \
  --warmup-rounds 3 \
  --hurst-samples 3 \
  --hurst-window-size 256 \
  --hurst-step-size 128 \
  --save-table outputs/ranked_table.csv \
  --save-plot outputs/ranked_frequencies.png \
  --save-trajectory-plot outputs/ranked_trajectory.png \
  --save-hurst-table outputs/ranked_hurst.csv \
  --save-hurst-plot outputs/ranked_hurst.png
```

## Step 1: Small-System Absorption Sweep

This runs many initial wealth distributions for a small system and saves one row per simulation:

```bash
MPLCONFIGDIR=.mplconfig python src/gamblers_ruin.py \
  --gamblers 6 \
  --total-wealth 80 \
  --pairing random \
  --sweep-vectors-per-family 3 \
  --sweep-sims-per-vector 20 \
  --small-sweep-output outputs/small_sweep_rows.csv \
  --small-sweep-summary outputs/small_sweep_summary.csv \
  --small-sweep-plot outputs/small_sweep_plot.png
```

The sweep output includes:

```text
initial amounts
HHI
1 - HHI
sum_{i < j} A_i A_j
Gini
max share
entropy
observed absorption time
absorbed/censored flag
winner
```

For the first small-system study, keep `--max-rounds 0`, which means no cap. If you set a finite `--max-rounds`, rows that do not absorb by the cap are right-censored and marked with `absorbed = False`.

Useful first test:

```bash
MPLCONFIGDIR=.mplconfig python src/gamblers_ruin.py \
  --gamblers 4 \
  --total-wealth 40 \
  --sweep-vectors-per-family 2 \
  --sweep-sims-per-vector 10 \
  --small-sweep-output outputs/smoke_sweep_rows.csv \
  --small-sweep-summary outputs/smoke_sweep_summary.csv \
  --small-sweep-plot outputs/smoke_sweep_plot.png
```

## Save An Animation

For saved animations, set `MPLCONFIGDIR=.mplconfig` so Matplotlib writes its cache inside this repo.

```bash
MPLCONFIGDIR=.mplconfig python src/gamblers_ruin.py --amounts 50,25,15,10 --sims 1000 --seed 7 --animate --output outputs/run.gif
```

Generated gambler animation:

```bash
MPLCONFIGDIR=.mplconfig python src/gamblers_ruin.py --gamblers 12 --total-wealth 300 --amount-mode descending --sims 500 --animate --output outputs/generated.gif
```

Use `.mp4` or `.m4v` instead of `.gif` if your local Matplotlib/ffmpeg setup supports it:

```bash
MPLCONFIGDIR=.mplconfig python src/gamblers_ruin.py --gamblers 8 --total-wealth 200 --sims 500 --animate --output outputs/run.mp4
```

## Show An Interactive Animation

This opens a Matplotlib window if your Python environment has a GUI backend available.

```bash
python src/gamblers_ruin.py --amounts 50,25,15,10 --sims 100 --animate
```

## Useful Flags

- `--amounts`: comma-separated exact initial amounts; overrides generated amounts.
- `--gamblers`: number of generated gamblers; must be even in this first model.
- `--total-wealth`: total starting wealth for generated amounts.
- `--amount-mode`: `descending`, `equal`, or `random`.
- `--sims`, `--trials`: number of independent simulations.
- `--seed`: reproducible random seed; use `--seed -1` for unpredictable randomness.
- `--max-rounds`: safety cap for each simulation; default `0` means no cap.
- `--pairing`: `random` or `ranked-after-warmup`.
- `--warmup-rounds`: random rounds before ranked pairing begins.
- `--save-table`: save the summary table as a CSV file.
- `--save-plot`: save expected vs observed survivor frequencies as an image.
- `--save-trajectory-plot`: save one wealth trajectory line plot.
- `--save-hurst-table`: save windowed Hurst estimates as a CSV file.
- `--save-hurst-plot`: save a mean Hurst-over-time plot with a 10th-90th percentile band.
- `--hurst-samples`: number of sample trajectories used for Hurst analysis.
- `--hurst-window-size`: rounds per Hurst window.
- `--hurst-step-size`: stride between Hurst windows.
- `--small-sweep-output`: run the step-1 absorption sweep and save per-simulation rows.
- `--small-sweep-summary`: save one aggregate row per initial vector for the step-1 sweep.
- `--small-sweep-plot`: save step-1 absorption-time predictor plots.
- `--sweep-vectors-per-family`: number of generated initial vectors per distribution family.
- `--sweep-sims-per-vector`: number of simulations per initial vector.
- `--animate`: animate one sample trajectory after the repeated simulations.
- `--output`: save animation to `.gif`, `.mp4`, or `.m4v`.
- `--interval-ms`: animation frame interval.
