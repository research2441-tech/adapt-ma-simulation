import json
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve

R = json.load(open('./s1_raw.json'))
main = R['main']
cal, test = main[:600], main[600:]
VAR = ['full', 'no_conf', 'single_view', 'linear_time', 'no_timing', 'no_keyphase']
rng = np.random.default_rng(11)


def raw(rows, v, a, k, rep='rec0'):
    return np.array([a * r[rep][v][0] + (1 - a) * r[rep][v][2] + k * r[rep][v][1] for r in rows])


def icc_a1(x, y):
    """Two-way random, absolute agreement, single measures ICC(A,1)."""
    Y = np.stack([x, y], 1); n, k = Y.shape
    gm = Y.mean(); msr = k * ((Y.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((Y.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


params, table = {}, {}
Qc = np.array([r['Q'] for r in cal]); Qt = np.array([r['Q'] for r in test])
for v in VAR:
    best = (-1, None)
    alphas = [1.0] if v == 'no_keyphase' else np.round(np.arange(0, 1.01, 0.1), 2)
    kaps = [0.0] if v in ('no_timing', 'linear_time') else [0, 5, 10, 15, 20, 30, 40, 60]
    for a in alphas:
        for k in kaps:
            rho = -spearmanr(Qc, raw(cal, v, a, k))[0]
            if rho > best[0]:
                best = (rho, (float(a), float(k)))
    a, k = best[1]
    x = raw(cal, v, a, k); b1, b0 = np.polyfit(x, Qc, 1)
    s0 = b0 + b1 * raw(test, v, a, k); s1 = b0 + b1 * raw(test, v, a, k, 'rec1')
    rho = spearmanr(Qt, s0)[0]; mae = np.abs(Qt - s0).mean()
    icc = icc_a1(s0, s1); sem = np.std(s0 - s1, ddof=1) / np.sqrt(2)
    bs = []
    for _ in range(1000):
        i = rng.integers(0, len(Qt), len(Qt))
        bs.append((spearmanr(Qt[i], s0[i])[0], np.abs(Qt[i] - s0[i]).mean(), icc_a1(s0[i], s1[i])))
    bs = np.array(bs)
    ci = lambda j: [float(np.percentile(bs[:, j], 2.5)), float(np.percentile(bs[:, j], 97.5))]
    params[v] = dict(alpha=a, kappa=k, b0=float(b0), b1=float(b1))
    table[v] = dict(rho=float(rho), rho_ci=ci(0), mae=float(mae), mae_ci=ci(1), icc=float(icc), icc_ci=ci(2),
                    sem=float(sem), mdc95=float(1.96 * np.sqrt(2) * sem), resid_sd=float(np.std(Qt - s0, ddof=1)))
    print(v, params[v], {k_: (round(x_, 3) if isinstance(x_, float) else [round(z, 3) for z in x_]) for k_, x_ in table[v].items()})

# paired bootstrap difference full vs each variant (Spearman)
diffs = {}
sf = params['full']
for v in VAR[1:]:
    p = params[v]
    xf = raw(test, 'full', sf['alpha'], sf['kappa']); xv = raw(test, v, p['alpha'], p['kappa'])
    d = []
    for _ in range(2000):
        i = rng.integers(0, len(Qt), len(Qt))
        d.append(spearmanr(Qt[i], -xf[i])[0] - spearmanr(Qt[i], -xv[i])[0])
    d = np.array(d)
    diffs[v] = [float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(np.mean(d <= 0))]
    print('diff', v, [round(z, 4) for z in diffs[v]])

# per-fault sensitivity of the full score (partial Spearman with each severity)
S = np.array([r['sev'] for r in test])
xf = -raw(test, 'full', sf['alpha'], sf['kappa'])
per_fault = [float(spearmanr(-S[:, j], xf)[0]) for j in range(5)]
print('per fault', np.round(per_fault, 3))

# sweeps
sweep = {}
for key, rows in R['sweeps'].items():
    Q = np.array([r['Q'] for r in rows]); out = {}
    for v in ('full', 'no_conf', 'single_view'):
        if v not in rows[0]['rec0']:
            continue
        p = params[v]
        s = p['b0'] + p['b1'] * raw(rows, v, p['alpha'], p['kappa'])
        out[v] = dict(rho=float(spearmanr(Q, s)[0]), mae=float(np.abs(Q - s).mean()))
    y = np.array([r['valgus_deg'] > 10 for r in rows])
    out['fppa_auc'] = float(roc_auc_score(y, [r['rec0']['fppa'] for r in rows]))
    if 'fppa_vr' in rows[0]['rec0']:
        out['fppa_vr_auc'] = float(roc_auc_score(y, [r['rec0']['fppa_vr'] for r in rows]))
    sweep[key] = out
    print(key, {k: v for k, v in out.items()})

# FPPA flag on the main set: threshold by Youden on calibration rows, raw and view-referenced
fppa = {}
for key in ('fppa', 'fppa_vr'):
    yc = np.array([r['valgus_deg'] > 10 for r in cal]); fc = np.array([r['rec0'][key] for r in cal])
    fpr, tpr, thr = roc_curve(yc, fc); th = float(thr[np.argmax(tpr - fpr)])
    yt = np.array([r['valgus_deg'] > 10 for r in test]); ft = np.array([r['rec0'][key] for r in test])
    sens = float(((ft >= th) & yt).sum() / yt.sum()); spec = float(((ft < th) & ~yt).sum() / (~yt).sum())
    auc = float(roc_auc_score(yt, ft))
    bs = []
    for _ in range(1000):
        i = rng.integers(0, len(yt), len(yt)); bs.append(roc_auc_score(yt[i], ft[i]))
    fr, tr, _ = roc_curve(yt, ft)
    fppa[key] = dict(threshold=th, sens=sens, spec=spec, auc=auc,
                     auc_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                     prevalence=float(yt.mean()), roc=[fr.tolist(), tr.tolist()],
                     scatter=[[r['valgus_deg'] for r in test], ft.tolist()], yaw=[r['yaw'] for r in test])
    print(key, {k: v for k, v in fppa[key].items() if k not in ('roc', 'scatter', 'yaw')})
# scatter data for the figure
scat = dict(Q=Qt.tolist(), full=(sf['b0'] + sf['b1'] * raw(test, 'full', sf['alpha'], sf['kappa'])).tolist(),
            retest=(sf['b0'] + sf['b1'] * raw(test, 'full', sf['alpha'], sf['kappa'], 'rec1')).tolist())
json.dump(dict(params=params, table=table, diffs=diffs, per_fault=per_fault, sweep=sweep, fppa=fppa, scatter=scat),
          open('./s1.json', 'w'))
