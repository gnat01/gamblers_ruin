# Absorption Time And Initial Wealth Distribution

The main time-to-win quantity in this model is the **absorption time**:

```text
T = number of rounds until only one gambler has positive wealth
```

The survival probability question is relatively clean. With fair pairwise dollar transfers, gambler `i` wins with probability:

```text
A_i / sum_j A_j
```

The absorption-time question is subtler. It should depend strongly on the initial distribution of wealth, not just on the identity of the richest gambler.

## Basic Intuition

Absorption is slow when wealth is spread out.

Absorption is fast when wealth is already concentrated.

If one gambler starts with almost all the wealth, the process is already near the absorbing state. If many gamblers start with comparable amounts, the system has to wander for a long time before one stack captures everything.

## A Natural Predictor

A first theoretical handle is:

```text
P(A) = sum_{i < j} A_i A_j
```

Equivalently, if:

```text
S = sum_i A_i
```

then:

```text
P(A) = 1/2 * [S^2 - sum_i A_i^2]
```

This quantity is large when wealth is evenly spread and small when wealth is concentrated.

For the two-gambler case, the expected absorption time is exactly:

```text
E[T] = A_1 A_2
```

So `sum_{i < j} A_i A_j` is the most natural multi-gambler generalization to test.

## Connection To HHI

Let:

```text
p_i = A_i / S
```

The Herfindahl-Hirschman index of the initial wealth distribution is:

```text
HHI = sum_i p_i^2
```

Then:

```text
P(A) = S^2 / 2 * (1 - HHI)
```

So the same hypothesis can be stated as:

```text
absorption time grows with total wealth squared
and shrinks as initial wealth concentration increases
```

A simple empirical model to test is:

```text
E[T] ~= C(strategy, N) * S^2 * (1 - HHI)
```

or equivalently:

```text
E[T] ~= C(strategy, N) * sum_{i < j} A_i A_j
```

The coefficient `C(strategy, N)` should depend on the pairing rule and the number of gamblers.

## Why This Is Plausible

If wealth is evenly distributed, then:

```text
HHI ~= 1 / N
```

and:

```text
1 - HHI
```

is large. The process is far from absorption.

If one gambler dominates initially, then:

```text
HHI ~= 1
```

and:

```text
1 - HHI
```

is small. The process is already close to absorption.

For random walks, absorption times usually scale quadratically with the relevant capital scale. So if all initial amounts are multiplied by a constant, time-to-absorption should grow roughly like the square of that constant.

## Role Of Pairing Strategy

The predictor above is about the initial state. Pairing strategy should affect the coefficient and the distribution of absorption times.

For random pairing, gamblers interact diffusely. Rich gamblers may play poor, middle, or rich gamblers.

For `ranked-after-warmup`, after the warmup:

```text
rank 1 plays rank 2
rank 3 plays rank 4
rank 5 plays rank 6
...
```

This should change the trajectory shape and possibly the absorption-time distribution. Rich gamblers are forced to contest other rich gamblers instead of farming small stacks. Poor gamblers also tend to fight near peers.

The broad hypothesis remains:

```text
for fixed N and fixed total wealth,
more initially equal wealth should imply longer absorption times
```

but the strategy may change:

```text
mean absorption time
median absorption time
tail behavior
variance across simulations
```

## Empirical Program

For each run, record:

```text
N
total wealth S
initial HHI
initial Gini
sum_{i < j} A_i A_j
max_i A_i / S
pairing strategy
warmup rounds
absorption time
winner
```

Then compare absorption time against predictors:

```text
P(A) = sum_{i < j} A_i A_j
HHI
1 - HHI
Gini
max share
```

Useful response variables:

```text
mean absorption time
median absorption time
p90 absorption time
p99 absorption time
standard deviation of absorption time
```

## Initial Distributions To Sweep

Useful families of initial wealth distributions:

```text
equal
mild descending inequality
severe descending inequality
one whale plus many small gamblers
random Dirichlet-like splits
two rich gamblers plus a long tail
```

For fixed `N` and fixed `S`, these distributions should create a clear range of HHI values. That makes it possible to test whether absorption time is mainly governed by initial concentration.

## First Thesis To Test

A concise first thesis:

```text
For fixed N, fixed total wealth S, and fair pairwise dollar transfers,
absorption time is primarily controlled by initial wealth dispersion.
A strong predictor is sum_{i < j} A_i A_j,
equivalently S^2 * (1 - HHI).
```

The ranked-after-warmup strategy should be compared against random pairing by fitting the same predictor under both strategies and checking whether the slope, intercept, and residual variance change.

## Caveats

The multi-gambler process is not just a two-player gambler's ruin process. Pairing rules, odd numbers of active gamblers, and the sequence of eliminations matter.

So the formula:

```text
E[T] ~= C(strategy, N) * sum_{i < j} A_i A_j
```

should be treated as a modeling hypothesis, not a theorem yet.

The right next step is empirical: generate many initial distributions, measure their concentration, run many simulations per distribution, and plot absorption time against `sum_{i < j} A_i A_j` and `1 - HHI`.
