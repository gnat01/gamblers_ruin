# Multiplayer Gambler's Ruin On A Lattice

The well-mixed multiplayer gambler's ruin model removes geometry: any active gambler can be paired with any other active gambler. A lattice version is qualitatively different because interactions are local. This makes it a natural candidate for pattern formation.

## Basic Lattice Setup

Let gamblers occupy sites on an `L x L` grid.

```text
site x has wealth A_x(t)
A_x(t) = 0 means the site is dead or inactive
total wealth is conserved
```

At each update, neighboring active sites play a fair dollar-transfer game:

```text
winner gains 1
loser loses 1
```

The process can use different neighborhoods:

```text
Von Neumann: up, down, left, right
Moore: 8 surrounding neighbors
```

## Update Rules

The update rule matters.

### Random Edge Update

Pick a random neighboring active pair and let them play.

This is asynchronous and simple. It is close to an interacting particle system.

### Synchronous Local Matching

Each round, form a random matching of lattice edges so that no site plays more than once in the same round. Then all matched pairs play simultaneously.

This is closer to the current round-based simulator.

### Neighbor Choice

Each active site chooses a neighboring active site to play.

This is more strategic, but it creates conflict-resolution issues when many sites choose the same neighbor.

For a first lattice model, random edge update or synchronous local matching is cleaner.

## Key Modeling Choice: Are Dead Sites Barriers?

The most important question is what happens when a site reaches zero.

### Variant A: Dead Sites Are Barriers

Only adjacent positive-wealth sites can play. If a zero-wealth region separates two active regions, those regions can no longer interact.

This means the process may not end in one global survivor. Instead it may freeze into several isolated surviving sites or clusters:

```text
final state = multiple active islands separated by dead zones
```

This is the most pattern-formation-friendly version. It can produce clusters, holes, interfaces, and frozen spatial configurations.

### Variant B: Dead Sites Are Empty But Not Barriers

If dead sites are not barriers, then some other mechanism is needed to let survivors interact across space.

Examples:

```text
gamblers move by random walk
gamblers play when adjacent
dead gamblers disappear
```

This becomes a mobile particle model rather than a fixed-site lattice model.

For pattern formation, the barrier version is the cleaner first experiment.

## Expected Pattern Phenomena

Local interactions can create spatial structure that is absent from the well-mixed model.

Possible phenomena:

```text
wealth clusters
dead zones
fronts and interfaces
coarsening
fragmentation
frozen active islands
percolation-like active clusters
```

Rich local regions may grow by absorbing nearby poor regions. Zero-wealth sites can form holes. Boundaries between surviving clusters can move like noisy domain walls.

The model may shift from:

```text
well-mixed eventual monopoly
```

to:

```text
spatial coarsening and frozen plural survival
```

## Von Neumann vs Moore Neighborhoods

The neighborhood should affect pattern geometry.

### Von Neumann

Only four nearest neighbors interact.

Expected behavior:

```text
more lattice anisotropy
more axis-aligned interfaces
slower mixing
slower coarsening
more jagged clusters
```

### Moore

Eight neighbors interact.

Expected behavior:

```text
faster local mixing
smoother clusters
less lattice anisotropy
faster interface motion
possibly fewer frozen fragments
```

## Useful Measurements

To make the pattern-formation question quantitative, track both wealth fields and active/dead masks.

Useful observables:

```text
active-site density over time
number of active clusters
largest cluster fraction
cluster size distribution
interface length between active and dead sites
wealth autocorrelation
correlation length
Moran's I
structure factor S(k)
surviving cluster count
time to frozen state
```

For a pattern-formation background, the most natural spatial diagnostics are probably:

```text
cluster size distribution
correlation length
structure factor
interface length
```

## Useful Visuals

The main visual should be an animation of the wealth field:

```text
heatmap of A_x(t)
```

Additional views:

```text
binary active/dead mask
cluster labels
wealth histogram over time
cluster size distribution over time
structure factor over time
```

The heatmap should reveal whether wealth forms domains, filaments, islands, or fronts.

## Initial Conditions To Try

The first lattice experiments should compare several initial wealth fields:

```text
uniform wealth
small random perturbations around uniform wealth
random wealth field with fixed total wealth
one rich seed in a poor background
several rich seeds
spatially correlated initial wealth
gradient initial wealth
```

Spatially correlated initial conditions are especially relevant for pattern formation because they let us ask whether initial patches persist, dissolve, or sharpen.

## Strategy Variants

Once the basic local model works, pairing can become strategic.

Examples:

```text
random neighbor
richest neighbor
poorest neighbor
nearest wealth neighbor
ranked local matching
boundary-only contests
```

The ranked-after-warmup idea has a local analogue:

```text
after W rounds, pair locally by wealth rank subject to lattice adjacency
```

This may sharpen fronts because rich boundary sites repeatedly contest other rich boundary sites.

## Main Hypotheses

Initial hypotheses:

```text
local interaction produces spatial correlations absent from the well-mixed model
dead-site barriers create frozen plural-survivor states
Von Neumann neighborhoods coarsen more slowly than Moore neighborhoods
high initial wealth dispersion nucleates dominant rich clusters earlier
equal initial wealth produces noisier coarsening and longer-lived patchiness
```

The most important conceptual change is:

```text
well-mixed gambler's ruin: eventual monopoly
lattice gambler's ruin: coarsening, fragmentation, and possible frozen domains
```

## First Design Decision

Before coding, decide:

```text
Do dead sites block interaction?
```

If yes, the model becomes a spatial coarsening and freezing process.

If no, the model needs mobility or long-range interaction rules to reach global absorption.

For pattern formation, the first model should probably use:

```text
fixed lattice
dead sites are barriers
synchronous random local matching
wealth heatmap animation
cluster and correlation diagnostics
```

This is the simplest version likely to show visible patterns.

## V0 Decision

The implemented v0 uses:

```text
fixed square lattice
dead sites are barriers
synchronous random local matching
Von Neumann or Moore neighborhoods
frozen islands as the natural final state
```

This means the model does not force a single global survivor. Instead, it naturally produces:

```text
wealth clusters
dead zones
fronts and interfaces
coarsening
fragmentation
frozen active islands
percolation-like active clusters
```

The implementation is:

```text
src/gamblers_ruin_square_lattice.py
```

Run details are in:

```text
docs/how_to_run.md
```

## Ways To Run V0

The main script is:

```text
src/gamblers_ruin_square_lattice.py
```

The most useful first run saves the actual spatial pattern:

```bash
MPLCONFIGDIR=.mplconfig python src/gamblers_ruin_square_lattice.py \
  --N 40 \
  --neighborhood neumann \
  --initial-mode random-gamma \
  --heterogeneity 5.0 \
  --initial-wealth 25 \
  --max-rounds 3000 \
  --frame-every 20 \
  --metric-every 10 \
  --save-animation outputs/lattice_neumann.gif \
  --save-heatmaps outputs/lattice_neumann_heatmaps.png \
  --save-wealth-histogram outputs/lattice_neumann_histogram.png \
  --save-cluster-size-table outputs/lattice_neumann_cluster_sizes.csv \
  --save-cluster-size-plot outputs/lattice_neumann_cluster_sizes.png \
  --save-metrics outputs/lattice_neumann_metrics.csv \
  --save-metrics-plot outputs/lattice_neumann_metrics.png \
  --save-final-summary outputs/lattice_neumann_summary.csv
```

Change the neighborhood with:

```text
--neighborhood neumann
--neighborhood moore
```

Change the initial field with:

```text
--initial-mode uniform
--initial-mode random-gamma
--initial-mode gradient
--initial-mode seeds
```

Use larger `--initial-wealth` values for coarsening and visible fronts. Very small values create rapid death-zone fragmentation.

The cluster-size outputs are especially important for the percolation/coarsening question:

```text
--save-cluster-size-table
--save-cluster-size-plot
```

The plot includes a log-log panel of:

```text
number of clusters of size s versus s
```

If `--sims` is greater than one, the log-log panel aggregates across simulations: the red curve is the mean count for each cluster size and the black band is the 90% interval. The plot then fits two log-log regimes: a blue dashed fit for small clusters with sizes 1-10 and a purple dashed fit for larger clusters above 10. This is better than a single naive line because the distribution often mixes many small fragments with a separate giant-cluster tail.

## Bifurcation Parameter

The bifurcation parameter should be the measured initial HHI:

```text
HHI = sum_x (A_x / S)^2
```

This is better than using a generator parameter such as gamma alpha because HHI is the actual initial wealth concentration of the realized lattice.

Interpretation:

```text
low HHI  = diffuse/equal initial wealth
high HHI = concentrated initial wealth
```

The code uses target HHI values to construct initial lattices, but it records and plots measured HHI:

```text
target_hhi
measured_target_hhi
initial_hhi
initial_one_minus_hhi
initial_gini
initial_max_share
```

The main scan is:

```bash
MPLCONFIGDIR=.mplconfig python src/gamblers_ruin_square_lattice.py \
  --N 32 \
  --neighborhood neumann \
  --initial-wealth 25 \
  --max-rounds 2000 \
  --sims 5 \
  --bifurcation-hhi-min 0.01 \
  --bifurcation-hhi-max 0.50 \
  --bifurcation-hhi-step 0.01 \
  --bifurcation-bins 8 \
  --bifurcation-output outputs/lattice_bifurcation.png \
  --bifurcation-table outputs/lattice_bifurcation.csv
```

The plotted responses are:

```text
final active density
final cluster count
largest island fraction
time to freeze or cap
```

Each panel includes individual runs as blue points, a red binned mean curve, and a black 90% interval band. The bin count is controlled by:

```text
--bifurcation-bins
```

This asks whether the final spatial pattern changes structurally as the initial wealth field moves from diffuse to concentrated.

Feasibility constraints:

```text
minimum HHI = 1 / (N*N)
high target HHI requires enough total wealth
integer wealth prevents exact targeting
```

Therefore the measured `initial_hhi` column is the value to use in analysis.
