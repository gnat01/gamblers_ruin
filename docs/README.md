# Random-Pairing Gambler's Ruin

First model:

- `N >= 2` gamblers, with `N` even
- gambler `i` starts with amount `A_i`
- each round shuffles the active gamblers and pairs them
- each pair flips a fair coin
- winner gains `$1`, loser loses `$1`
- gamblers at `$0` are no longer paired
- the process stops when one gambler owns all wealth

For the fair game, gambler `i` wins with probability:

```text
A_i / sum_j A_j
```

So the initially richest gambler should win most often across repeated experiments, but not always.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run Repeated Experiments

```bash
python gamblers_ruin.py --amounts 50,25,15,10 --trials 1000 --seed 7
```

## Save One Animation

```bash
MPLCONFIGDIR=.mplconfig python gamblers_ruin.py --amounts 50,25,15,10 --trials 1000 --seed 7 --animate --output outputs/run.gif
```

Use `--seed -1` for unpredictable randomness.
