"""Compare small linear and quadratic models for next-day ITA returns."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def features(prices):
    """Row t uses returns through close t; its target is return t to t+1."""
    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 1 or len(prices) < 23 or not np.all(np.isfinite(prices) & (prices > 0)):
        raise ValueError('Need at least 23 finite positive prices.')
    returns = prices[1:] / prices[:-1] - 1
    t = np.arange(20, len(prices)-1)
    x = np.array([[returns[i-1], returns[i-5:i].mean(),
                   returns[i-20:i].mean()] for i in t])
    return x, returns[t], t+1


def design(x, quadratic=False):
    # No interaction terms: just an intercept, three inputs, and their squares.
    return np.column_stack([np.ones(len(x)), x, x*x] if quadratic
                           else [np.ones(len(x)), x])


def fit_predict(x, y, fit_mask, predict_mask, quadratic=False):
    # Scaling uses fitting data only; it improves numerical conditioning.
    mean, scale = x[fit_mask].mean(0), x[fit_mask].std(0)
    scale = np.where(scale > 0, scale, 1)
    matrix = design((x-mean)/scale, quadratic)
    coef = np.linalg.lstsq(matrix[fit_mask], y[fit_mask], rcond=None)[0]
    return matrix[predict_mask] @ coef, dict(coefficients=coef.tolist(),
                                           mean=mean.tolist(), scale=scale.tolist())


def rmse(y, prediction):
    return float(np.sqrt(np.mean((y-prediction)**2)))


def run(source, output):
    with source.open() as f:
        rows = list(csv.DictReader(f))
    dates = np.array([r['Date'] for r in rows])
    if not np.all(dates[1:] > dates[:-1]):
        raise ValueError('Dates must be unique and sorted.')
    x, y, indices = features([float(r['Close']) for r in rows])
    dates = dates[indices]
    train = dates < '2018-01-01'
    validation = (dates >= '2018-01-01') & (dates < '2022-01-01')
    test = dates >= '2022-01-01'
    if min(train.sum(), validation.sum(), test.sum()) < 100:
        raise ValueError('Need at least 100 targets in each period.')
    scores = {}
    for name in ['linear', 'quadratic']:
        pred, _ = fit_predict(x,y,train,validation,name=='quadratic')
        scores[name] = rmse(y[validation],pred)
    selected = min(scores, key=scores.get)
    predictions = {}; models = {}; test_scores = {}
    for name in scores:
        pred, model = fit_predict(x,y,train|validation,test,name=='quadratic')
        predictions[name] = pred; models[name] = model
        test_scores[name] = rmse(y[test],pred)
    baseline = rmse(y[test],np.zeros(test.sum()))
    output.mkdir(parents=True,exist_ok=True)
    result = dict(selected_on_validation=selected,validation_rmse=scores,
        test_rmse=test_scores,zero_return_baseline_rmse=baseline,
        test_observations=int(test.sum()),test_start=dates[test][0],test_end=dates[test][-1],
        improvement_pct={k:100*(1-v/baseline) for k,v in test_scores.items()},
        models=models,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        limitation='Retrospective comparison on a later period already explored in other experiments.')
    (output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    with (output/'predictions.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['target_date','actual_return','linear','quadratic','baseline'])
        w.writerows(zip(dates[test],y[test],predictions['linear'],predictions['quadratic'],np.zeros(test.sum())))
    print(json.dumps({k:v for k,v in result.items() if k!='models'},indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('results/regression'))
    args=parser.parse_args()
    run(args.data,args.output)
