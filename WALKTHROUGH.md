# Reading the experiment

Start with `kalman` in `fair_value.py`. It is the whole model. `simulate` generates the observations; `ewma` is its closest simple competitor. The remaining functions run comparisons and save charts.

## One update by hand

Suppose the last estimate is 100 and its uncertainty (variance) is 1. The value can move between observations, so add process variance Q = 1. The predicted uncertainty is now 2.

The next quote is 102 and observation variance R = 1. The weight is 2 / (2 + 1), or two-thirds. The corrected estimate is 100 + two-thirds × (102 − 100) = 101.333. Its variance falls to (1 − two-thirds) × 2 = 0.667.

The first automated test checks these exact numbers.

## What Q and R mean

- Larger Q: the underlying value can change more between quotes. Respond faster.
- Larger R: observations are noisier. Smooth them more heavily.
- Variance is squared price units; standard deviation is its square root.

Q and R are assumed known here. The filter learns the current state from observations; it does not learn those parameters in this experiment.

## Why compare with a moving average?

EWMA uses the same correction formula with a fixed weight. Under steady conditions, Kalman's weight settles too. A complicated name does not imply a more useful estimate. The comparison identifies situations where tracking uncertainty changes behaviour, especially after missing observations.

## How to interpret the chart

Black is the simulated hidden value. Grey dots are observations. Blue and orange are estimates using only information available at that time. A smoother curve is not automatically better: after a sudden jump, excessive smoothing creates lag.

The blue band widens when quotes disappear. It is uncertainty under the assumed model, not a guarantee. In the jump scenario the assumptions are deliberately wrong.

## What the result supports

The experiment shows how noise, jumps and missing observations affect a causal estimator. It does not show that real assets have an observable true value, that these settings are optimal, or that trading the estimate makes money.

To reproduce the main conclusion, run the script and compare the errors. To explore the mechanism, change one scenario's `process_std` or `quote_std`, rerun into a new directory, and predict the effect before looking at the chart.
