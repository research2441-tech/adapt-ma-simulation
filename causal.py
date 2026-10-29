"""Experiment S4: dose-response (instrumental variable) and interventional mediation estimators
under clustering, unmeasured motivation and an exposure-induced mediator-outcome confounder."""
import json, sys
import numpy as np
from scipy import stats


def clusters(K, rng, mbar=45, cv=0.4):
    shape = 1 / cv ** 2
    m = np.maximum(10, rng.gamma(shape, mbar / shape, K).round().astype(int))
    z = np.zeros(K, int); z[rng.permutation(K)[:K // 2]] = 1
    cl = np.repeat(np.arange(K), m)
    return cl, z[cl], len(cl)


def ols(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


def cr_se(X, resid, cl, K, bread_X=None):
    """CR1 cluster-robust standard errors."""
    Xb = X if bread_X is None else bread_X
    B = np.linalg.inv(Xb.T @ Xb)
    S = np.zeros((X.shape[1], X.shape[1]))
    G = np.zeros((K, X.shape[1])); np.add.at(G, cl, X * resid[:, None])
    S = G.T @ G
    n, p = X.shape
    c = K / (K - 1) * (n - 1) / (n - p)
    return np.sqrt(np.diag(c * B @ S @ B))


BETA_DOSE = 1.6


def sim_dose(K, rng):
    cl, z, n = clusters(K, rng)
    U = rng.normal(0, 1, n)
    wj = rng.normal(0, 1, K)[cl]
    D = np.where(z == 1, 1 / (1 + np.exp(-(0.9 + 0.9 * U + 0.3 * wj + rng.normal(0, 0.5, n)))), 0.0)
    u0 = rng.normal(0, 1.3, K)[cl]; u1 = 0.8 * u0 + 0.6 * rng.normal(0, 1.3, K)[cl]
    y0 = 44 + 1.5 * U + u0 + rng.normal(0, 5, n)
    y1 = 44.6 + 0.7 * (y0 - 44) + BETA_DOSE * D + 1.0 * U + u1 + rng.normal(0, 3.5, n)
    one = np.ones(n); yc = y0 - y0.mean()
    out = {}
    # (a) naive dose-response within the intervention arm
    t = z == 1
    Xa = np.column_stack([one[t], D[t], yc[t]]); ba = ols(Xa, y1[t])
    sa = cr_se(Xa, y1[t] - Xa @ ba, cl[t], K)
    out['naive'] = (ba[1], sa[1], int(np.unique(cl[t]).size) - 2)
    # (b) dose-response across arms ignoring randomisation (as-treated regression)
    Xb = np.column_stack([one, D, yc]); bb = ols(Xb, y1); sb = cr_se(Xb, y1 - Xb @ bb, cl, K)
    out['as_treated'] = (bb[1], sb[1], K - 2)
    # (c) two-stage least squares with randomisation as instrument
    Z = np.column_stack([one, z, yc]); g = ols(Z, D); Dh = Z @ g
    X2 = np.column_stack([one, Dh, yc]); b2 = ols(X2, y1)
    res = y1 - np.column_stack([one, D, yc]) @ b2
    s2 = cr_se(X2, res, cl, K)
    out['iv'] = (b2[1], s2[1], K - 2)
    return out


IIE_TRUE = 0.9 * (0.30 + 0.05 * 4.0)


def fit_mediation(cl, z, y0, L, M, y1):
    n = len(z); one = np.ones(n); yc = y0 - y0.mean()
    # interventional indirect effect: outcome model adjusts for the exposure-induced confounder L;
    # the mediator contrast is taken marginally over L (random draw of M under each arm)
    by = ols(np.column_stack([one, z, L, M, yc]), y1)
    bm = ols(np.column_stack([one, z, yc]), M)
    iie = by[3] * bm[1]
    # naive product of coefficients without L
    by_n = ols(np.column_stack([one, z, M, yc]), y1)
    naive = by_n[2] * bm[1]
    # naive product adjusting L in both models (treats L as a pre-treatment confounder)
    bm_l = ols(np.column_stack([one, z, L, yc]), M)
    naive_l = by[3] * bm_l[1]
    return np.array([iie, naive, naive_l])


def sim_med(K, rng, B=200):
    cl, z, n = clusters(K, rng)
    uj = rng.normal(0, 1, K)
    L = 4.0 * z + 1.5 * uj[cl] * 0.3 + rng.normal(0, 6, n)          # change in MVPA (min/day)
    M = 0.30 * z + 0.05 * L + 0.15 * uj[cl] + rng.normal(0, 1, n)   # change in motor competence (z units)
    y0 = 44 + rng.normal(0, 1.3, K)[cl] + rng.normal(0, 5, n)
    y1 = 44.6 + 0.7 * (y0 - 44) + 0.5 * z + 0.08 * L + 0.9 * M + rng.normal(0, 1.0, K)[cl] + rng.normal(0, 3.5, n)
    est = fit_mediation(cl, z, y0, L, M, y1)
    # cluster bootstrap stratified by arm
    ids1 = np.unique(cl[z == 1]); ids0 = np.unique(cl[z == 0])
    idx = [np.where(cl == j)[0] for j in range(K)]
    bs = []
    for _ in range(B):
        pick = np.concatenate([rng.choice(ids1, len(ids1)), rng.choice(ids0, len(ids0))])
        rows = np.concatenate([idx[j] for j in pick])
        newcl = np.concatenate([np.full(len(idx[j]), k) for k, j in enumerate(pick)])
        bs.append(fit_mediation(newcl, z[rows], y0[rows], L[rows], M[rows], y1[rows]))
    bs = np.array(bs)
    lo, hi = np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
    return est, lo, hi


if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    res = {'beta_dose': BETA_DOSE, 'iie_true': IIE_TRUE, 'dose': {}, 'med': {}}
    for K in (16, 24, 32):
        rng = np.random.default_rng(100 + K)
        acc = {k: [] for k in ('naive', 'as_treated', 'iv')}
        for _ in range(reps):
            o = sim_dose(K, rng)
            for k, (b, s, df) in o.items():
                q = stats.t.ppf(0.975, df)
                acc[k].append((b, s, (b - q * s <= BETA_DOSE <= b + q * s)))
        res['dose'][K] = {k: dict(bias=float(np.mean([a[0] for a in v]) - BETA_DOSE),
                                  rmse=float(np.sqrt(np.mean([(a[0] - BETA_DOSE) ** 2 for a in v]))),
                                  emp_se=float(np.std([a[0] for a in v])), mean_se=float(np.mean([a[1] for a in v])),
                                  cover=float(np.mean([a[2] for a in v]))) for k, v in acc.items()}
        print('dose', K, res['dose'][K], flush=True)
    mreps = max(200, reps // 2)
    for K in (16, 24, 32):
        rng = np.random.default_rng(900 + K)
        E, C = [], []
        for _ in range(mreps):
            e, lo, hi = sim_med(K, rng)
            E.append(e); C.append((lo <= IIE_TRUE) & (IIE_TRUE <= hi))
        E = np.array(E); C = np.array(C)
        res['med'][K] = {name: dict(bias=float(E[:, i].mean() - IIE_TRUE), rmse=float(np.sqrt(((E[:, i] - IIE_TRUE) ** 2).mean())),
                                    emp_se=float(E[:, i].std()), cover=float(C[:, i].mean()))
                         for i, name in enumerate(['interventional', 'naive_noL', 'naive_adjL'])}
        print('med', K, res['med'][K], flush=True)
    json.dump(res, open('./s4.json', 'w'))
