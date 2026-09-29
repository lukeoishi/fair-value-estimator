# ITA return forecasting: linear versus quadratic

A small experiment asking whether recent ITA returns help predict the next trading day's return.

## Model

Three inputs, all available at today's close:

- Today's simple return.
- Mean return over the last five trading days, including today.
- Mean return over the last twenty trading days, including today.

Linear regression fits an intercept and three coefficients. The quadratic version also includes each input's square: seven coefficients in total, with no interactions. Both use ordinary least squares. Inputs are standardised using fitting data only. There are no neural networks or tuning libraries.

## Evaluation

Fit on targets before 2018, compare models on 2018–2021, then select the lower validation RMSE. Refit each model through 2021 and freeze its coefficients for the 2022–17 September 2026 comparison. Daily inputs update, but test targets never enter fitting. The baseline predicts a zero return (tomorrow's close equals today's).

| Model | Validation RMSE | Later-period RMSE |
|---|---:|---:|
| Linear | 1.8242% | 1.2803% |
| Quadratic | 1.8162% | 1.2931% |
| Zero return | — | 1.2813% |

Quadratic regression won validation but lost to the baseline on the 1,181 later observations. Linear regression's later RMSE was only 0.073% lower than the baseline; this is not evidence of a reliable forecasting advantage. The selected quadratic model was 0.922% worse. RMSE measures return prediction error, not profit or directional accuracy.

The later period has already been examined in earlier project experiments. This is an exploratory retrospective comparison, not a fresh untouched holdout. No transaction costs or trading strategy are tested. The adjusted price snapshot came from the earlier project's Yahoo/yfinance cache; it is not independently verified point-in-time data. Its hash is recorded in the results.

## Reproduce

From the repository directory:

```sh
python regression.py --data /path/to/ita_daily_adjusted.csv
python -m unittest discover -s tests
```

The CSV needs sorted, unique `Date` and positive adjusted `Close` columns spanning the stated periods. The complete raw snapshot is not included. Outputs are `results/regression/metrics.json` (scores, scaling and fitted coefficients) and `predictions.csv` (every later-period prediction). `regression.py` contains the whole experiment. Tests check feature/target timing, independence from future prices and recovery of a known quadratic relationship.
