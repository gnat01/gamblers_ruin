# Empirical Sweep Plan For Absorption Time

The next empirical step is to generate many initial wealth distributions, measure their concentration, run many simulations per distribution, and plot absorption time against:

```text
sum_{i < j} A_i A_j
```

and:

```text
1 - HHI
```

This should test whether absorption time is primarily controlled by the initial spread of wealth.

## Core Experimental Object

Each experimental condition should specify:

```text
N
S = total wealth
initial wealth vector A
pairing strategy
warmup rounds
number of simulations per A
```

For each initial wealth vector `A`, compute predictors:

```text
pairwise_product_sum = sum_{i < j} A_i A_j
HHI = sum_i (A_i / S)^2
1 - HHI
Gini
max_share = max_i A_i / S
entropy
```

Then run many simulations from that same `A` and record:

```text
mean absorption time
median absorption time
p90 absorption time
standard deviation of absorption time
capped fraction, if using max_rounds
winner frequencies
```

## First Empirical Question

For fixed `N`, fixed `S`, and fixed pairing strategy:

```text
Does mean absorption time scale roughly linearly with sum_{i < j} A_i A_j?
```

Since:

```text
sum_{i < j} A_i A_j = S^2 / 2 * (1 - HHI)
```

for fixed `S`, plotting against `sum_{i < j} A_i A_j` and plotting against `1 - HHI` are essentially two views of the same information.

Both are useful:

```text
sum_{i < j} A_i A_j
```

connects directly to the two-gambler formula:

```text
E[T] = A_1 A_2
```

while:

```text
1 - HHI
```

connects directly to concentration.

## Initial Distribution Families

We need a wide spread of wealth shapes, not just random noise.

Useful families:

```text
equal
linear descending
geometric descending
one whale plus flat tail
two whales plus flat tail
Dirichlet random splits with different alpha values
```

Dirichlet-style splits are especially useful:

```text
large alpha -> near equal wealth
alpha = 1   -> moderate random inequality
small alpha -> spiky unequal wealth
```

After generating proportions, convert them to integer wealth values summing to `S`.

## Important Design Choice

Keep `N` and `S` fixed at first.

Example:

```text
N = 8
S = 160
strategy = random
simulations_per_distribution = 100
```

Varying `N`, `S`, wealth shape, and strategy all at once would confound too many things.

The first target should be:

```text
fixed N
fixed S
many initial wealth vectors
one pairing strategy
```

Then repeat the exact same wealth vectors under:

```text
strategy = ranked-after-warmup
warmup_rounds = 3
```

Using the same initial vectors across strategies is important because it makes the strategy comparison paired rather than noisy.

## Plots To Produce

### Pairwise Product Scatter

```text
x = sum_{i < j} A_i A_j
y = mean absorption time
color = distribution family
```

This tests the direct multi-gambler analogue of the two-player result.

### Normalized HHI Plot

```text
x = 1 - HHI
y = mean absorption time / S^2
color = distribution family
```

This is the cleaner theoretical plot when comparing across different total wealth values later.

### Median And Tail Plot

Absorption time can be noisy and heavy-tailed, so do not rely only on the mean.

Plot:

```text
x = pairwise_product_sum
y = median absorption time
error band = p10 to p90
```

or:

```text
x = 1 - HHI
y = median absorption time / S^2
error band = p10 to p90
```

### Strategy Comparison Plot

Use the same initial wealth vectors under both strategies:

```text
random
ranked-after-warmup
```

Plot both fitted lines:

```text
x = pairwise_product_sum
y = mean absorption time
color = strategy
```

This tests whether ranked pairing changes the slope, intercept, or variance around the basic concentration predictor.

### Residual Plot

Fit a simple model using the pairwise product predictor, then plot:

```text
x = predictor or fitted value
y = actual absorption time - predicted absorption time
color = distribution family
```

This tells us whether Gini, max share, or distribution family adds information beyond `HHI`.

## First Regressions

Start simple:

```text
mean_T ~ beta_0 + beta_1 * pairwise_product_sum
```

Then normalized:

```text
mean_T / S^2 ~ beta_0 + beta_1 * (1 - HHI)
```

Then test extra predictors:

```text
mean_T / S^2 ~ beta_0
             + beta_1 * (1 - HHI)
             + beta_2 * max_share
             + beta_3 * Gini
```

If `1 - HHI` explains most of the variation, that is a strong story.

If `max_share` or `Gini` adds substantial explanatory power, then the initial wealth distribution has structure not captured by HHI alone.

## Heavy-Tail Issue

Absorption time may have a heavy tail.

Therefore, report:

```text
mean
median
p90
p99
standard deviation
capped fraction
```

If the mean is unstable but the median is clean, that is itself an important finding.

If the p90 or p99 behaves differently from the mean, then the theory should distinguish typical absorption time from tail absorption time.

## Suggested First Experiment

Start modestly:

```text
N = 8
S = 160
initial vectors = 30
simulations per vector = 50
strategies = random, ranked-after-warmup
warmup_rounds = 3
```

This gives:

```text
30 * 50 * 2 = 3000 simulations
```

That is enough to validate the pipeline and see whether the scatterplot is structured.

If the relationship is visible, scale up:

```text
N = 8
S = 160
initial vectors = 60
simulations per vector = 100
strategies = random, ranked-after-warmup
```

This gives:

```text
60 * 100 * 2 = 12000 simulations
```

The first goal is not final truth. The first goal is to see whether absorption time has a clear empirical relationship with initial wealth concentration.

## Expected Thesis

The first thesis to test:

```text
For fixed N, fixed total wealth S, and fair pairwise dollar transfers,
absorption time is primarily controlled by initial wealth dispersion.
The strongest first predictor is sum_{i < j} A_i A_j,
equivalently S^2 * (1 - HHI).
```

The ranked-after-warmup strategy should then be compared against random pairing by checking whether it changes:

```text
slope
intercept
residual variance
tail behavior
```

under the same predictor.
