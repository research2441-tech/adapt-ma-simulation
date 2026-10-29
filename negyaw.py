import json, numpy as np
from run_mqs import run
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
S1 = json.load(open('./s1.json')); params = S1['params']
def raw(rows, v, a, k, rep='rec0'):
    return np.array([a * r[rep][v][0] + (1 - a) * r[rep][v][2] + k * r[rep][v][1] for r in rows])
out = {}
for y in (-60, -50, -40, -30, -20, -10):
    rows = run(400, yaw_fixed=float(y), variants=['full', 'single_view'], seed0=500_000 + abs(y) * 100)
    Q = np.array([r['Q'] for r in rows]); o = {}
    for v in ('full', 'single_view'):
        p = params[v]; s = p['b0'] + p['b1'] * raw(rows, v, p['alpha'], p['kappa'])
        o[v] = dict(rho=float(spearmanr(Q, s)[0]), mae=float(np.abs(Q - s).mean()))
    yy = np.array([r['valgus_deg'] > 10 for r in rows])
    o['fppa_auc'] = float(roc_auc_score(yy, [r['rec0']['fppa'] for r in rows]))
    o['fppa_vr_auc'] = float(roc_auc_score(yy, [r['rec0']['fppa_vr'] for r in rows]))
    out['yaw_%d' % y] = o; print(y, o, flush=True)
json.dump(out, open('./s1_negyaw.json', 'w'))
