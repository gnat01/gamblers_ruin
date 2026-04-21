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
- `--save-table`: save the summary table as a CSV file.
- `--save-plot`: save expected vs observed survivor frequencies as an image.
- `--animate`: animate one sample trajectory after the repeated simulations.
- `--output`: save animation to `.gif`, `.mp4`, or `.m4v`.
- `--interval-ms`: animation frame interval.
