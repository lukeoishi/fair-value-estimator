"""Recent-weighted sine-wave forecasts. No trend term or future observations."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def forecast(prices, periods=(60,), half_life=30):
    """Fit c + sine/cosine terms; return the forecast one trading step ahead.

    A sine/cosine pair is equivalent to one phase-shifted sine wave.
    Periods are measured in trading days. Weights halve every half_life days.
    """
    prices = np.asarray(prices, dtype=float)
    if (prices.ndim != 1 or len(prices) < 10 or
            not np.isfinite(prices).all() or np.any(prices <= 0)):
        raise ValueError('Provide at least ten finite positive prices.')
    if half_life <= 0 or not np.isfinite(half_life):
        raise ValueError('Half-life must be positive and finite.')
    if not periods or any(not np.isfinite(p) or p <= 2 for p in periods):
        raise ValueError('Periods must exceed two trading days.')
    if len(set(periods)) != len(periods):
        raise ValueError('Periods must be distinct.')
    t = np.arange(len(prices) + 1, dtype=float)
    columns = [np.ones_like(t)]
    for period in periods:
        columns.extend([np.sin(2 * np.pi * t / period),
                        np.cos(2 * np.pi * t / period)])
    design = np.column_stack(columns)
    weights = 2.0 ** (-(len(prices) - 1 - t[:-1]) / half_life)
    # Weighted least squares uses square-root weights on both sides.
    root = np.sqrt(weights)
    coefficients = np.linalg.lstsq(design[:-1] * root[:, None],
                                    prices * root, rcond=None)[0]
    return float(design[-1] @ coefficients)


def walk_forward(prices, periods, half_life, window=120):
    prediction = np.full(len(prices), np.nan)
    for t in range(window, len(prices)):
        prediction[t] = forecast(prices[t-window:t], periods, half_life)
    return prediction


def run(source, output):
    with source.open() as handle:
        rows = list(csv.DictReader(handle))
    dates = np.array([r['Date'] for r in rows])
    prices = np.array([float(r['Close']) for r in rows])
    if not all(dates[i] < dates[i+1] for i in range(len(dates)-1)):
        raise ValueError('Dates must be unique and increasing.')
    calibration = (dates >= '2018-01-01') & (dates < '2022-01-01')
    test = dates >= '2022-01-01'
    if calibration.sum() < 100 or test.sum() < 100:
        raise ValueError('Need at least 100 observations in each evaluation period.')
    previous = np.roll(prices, 1)
    trials = []
    for periods in [(60,), (120,), (20, 60)]:
        for half_life in [10, 30, 60]:
            predicted = walk_forward(prices, periods, half_life)
            error = (predicted - prices) / previous
            score = float(np.sqrt(np.mean(error[calibration] ** 2)))
            trials.append((score, periods, half_life, predicted))
    _, periods, half_life, predicted = min(trials, key=lambda row: row[0])
    e = ((predicted[test] - prices[test]) / previous[test]) ** 2
    b = ((previous[test] - prices[test]) / previous[test]) ** 2
    model_rmse, baseline_rmse = np.sqrt(e.mean()), np.sqrt(b.mean())
    # Paired moving-block bootstrap: uncertainty conditional on this model choice.
    rng = np.random.default_rng(77)
    n = len(e)
    starts = rng.integers(0, n-19, (2000, int(np.ceil(n/20))))
    indices = (starts[:, :, None] + np.arange(20)).reshape(2000, -1)[:, :n]
    improvements = 100 * (1 - np.sqrt(e[indices].mean(1) / b[indices].mean(1)))
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(periods=list(periods), half_life=half_life, window=120,
        calibration='2018–2021', test_start=dates[test][0], test_end=dates[test][-1],
        observations=int(test.sum()), model_rmse_pct=100*model_rmse,
        baseline_rmse_pct=100*baseline_rmse,
        improvement_pct=100*(1-model_rmse/baseline_rmse),
        improvement_95pct_ci=np.percentile(improvements,[2.5,97.5]).tolist(),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        metric='RMSE of next-close error divided by previous adjusted close',
        limitation='Retrospective test, already explored in previous projects; not an untouched holdout.')
    (output/'metrics.json').write_text(json.dumps(summary, indent=2)+'\n')
    with (output/'predictions.csv').open('w', newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['date','actual_adjusted_close','forecast','last_close_baseline'])
        writer.writerows(zip(dates[test],prices[test],predicted[test],previous[test]))
    with (output/'calibration.csv').open('w', newline='') as f:
        writer=csv.writer(f);writer.writerow(['periods','half_life','rmse'])
        writer.writerows((str(p),h,s) for s,p,h,_ in trials)
    fig, ax = plt.subplots(figsize=(11,5))
    selected=np.flatnonzero(test)[-120:]
    time=np.array(dates,dtype='datetime64[D]')[selected]
    ax.plot(time,prices[selected],label='Observed close',color='#222222')
    ax.plot(time,predicted[selected],label='One-day harmonic forecast',color='#2255cc')
    ax.plot(time,previous[selected],label='Last-close baseline',color='#bb7733',alpha=.65)
    ax.set(title='ITA: forecasts made before each observed close',ylabel='Adjusted price (USD)')
    ax.legend(frameon=False);fig.autofmt_xdate();fig.tight_layout()
    fig.savefig(output/'forecast.png',dpi=160);plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True,type=Path)
    parser.add_argument('--output',type=Path,default=Path('results/harmonic'))
    args=parser.parse_args()
    run(args.data,args.output)
