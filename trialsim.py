"""Experiment S3: operating characteristics of the primary analysis for the cluster trial.

Random-intercept ANCOVA fitted by profile REML using cluster sufficient statistics, so that
thousands of simulated trials can be analysed quickly. Four analyses are compared.
"""
import json, sys, time
import numpy as np
from scipy.optimize import minimize_scalar
from scipy import stats


def gen_trial(K, mbar, icc, delta, rng, sd=6.0, r_ind=0.7, r_clu=0.8, cv=0.4, attr=0.15, mu=44.0):
    """Simulate one two-arm cluster trial (K schools, 1:1) with baseline and follow-up VO2peak."""
    shape = 1 / cv ** 2
    m = np.maximum(10, rng.gamma(shape, mbar / shape, K).round().astype(int))
    z = np.zeros(K, int); z[rng.permutation(K)[:K // 2]] = 1
    su, se = np.sqrt(icc) * sd, np.sqrt(1 - icc) * sd
    u0 = rng.normal(0, su, K); u1 = r_clu * u0 + np.sqrt(1 - r_clu ** 2) * rng.normal(0, su, K)
    cl = np.repeat(np.arange(K), m); N = m.sum()
    e0 = rng.normal(0, se, N); e1 = r_ind * e0 + np.sqrt(1 - r_ind ** 2) * rng.normal(0, se, N)
    y0 = mu + u0[cl] + e0
    y1 = mu + 0.6 + delta * z[cl] + u1[cl] + e1
    # missing follow-up: MAR, more likely for low-fitness children
    pm = 1 / (1 + np.exp(-(np.log(attr / (1 - attr)) - 0.4 * (y0 - mu) / sd)))
    keep = rng.random(N) > pm
    return cl[keep], z[cl][keep], y0[keep], y1[keep], K


def reml_ri(cl, X, y, K):
    """Random-intercept LMM by profile REML. Returns beta, cov(beta), lambda, sigma2."""
    n_j = np.bincount(cl, minlength=K).astype(float)
    p = X.shape[1]
    Sx = np.zeros((K, p)); np.add.at(Sx, cl, X)
    Sy = np.bincount(cl, weights=y, minlength=K)
    XtX, Xty, yty = X.T @ X, X.T @ y, y @ y
    N = len(y)

    def parts(lam):
        w = lam / (1 + n_j * lam)
        A = XtX - (Sx * w[:, None]).T @ Sx
        b = Xty - (Sx * (w * Sy)[:, None]).sum(0)
        beta = np.linalg.solve(A, b)
        rss = yty - (w * Sy ** 2).sum() - beta @ b
        return A, beta, rss

    def nll(loglam):
        lam = np.exp(loglam)
        A, beta, rss = parts(lam)
        s2 = rss / (N - p)
        return 0.5 * ((N - p) * np.log(s2) + np.log1p(n_j * lam).sum() + np.linalg.slogdet(A)[1])

    res = minimize_scalar(nll, bounds=(-12, 3), method='bounded', options={'xatol': 1e-4})
    lam = np.exp(res.x)
    A, beta, rss = parts(lam)
    s2 = rss / (N - p)
    return beta, s2 * np.linalg.inv(A), lam, s2


def analyse(cl, z, y0, y1, K):
    yc = y0 - y0.mean()
    X = np.column_stack([np.ones_like(y1), z, yc])
    beta, V, lam, s2 = reml_ri(cl, X, y1, K)
    se = np.sqrt(V[1, 1]); t = beta[1] / se
    out = {}
    out['lmm_z'] = 2 * stats.norm.sf(abs(t))
    out['lmm_bw'] = 2 * stats.t.sf(abs(t), K - 2)                  # between-within df
    # cluster-level ANCOVA on cluster means of change (unweighted), t with K-3 df
    m = np.bincount(cl, minlength=K)
    ok = m > 0
    zc = np.bincount(cl, weights=z, minlength=K)[ok] / m[ok]
    d = (np.bincount(cl, weights=y1, minlength=K)[ok] / m[ok])
    b = (np.bincount(cl, weights=y0, minlength=K)[ok] / m[ok])
    Xc = np.column_stack([np.ones(ok.sum()), zc, b - b.mean()])
    bc, *_ = np.linalg.lstsq(Xc, d, rcond=None)
    r = d - Xc @ bc; dfc = ok.sum() - 3
    vc = (r @ r / dfc) * np.linalg.inv(Xc.T @ Xc)
    out['cluster'] = 2 * stats.t.sf(abs(bc[1] / np.sqrt(vc[1, 1])), dfc)
    # individual-level OLS ignoring clustering
    bo, *_ = np.linalg.lstsq(X, y1, rcond=None)
    ro = y1 - X @ bo; vo = (ro @ ro / (len(y1) - 3)) * np.linalg.inv(X.T @ X)
    out['ols'] = 2 * stats.norm.sf(abs(bo[1] / np.sqrt(vo[1, 1])))
    icc_hat = lam / (1 + lam)
    return out, float(beta[1]), float(se), float(icc_hat)


def cell(K, mbar, icc, delta, reps, seed):
    rng = np.random.default_rng(seed)
    P = {k: 0 for k in ('lmm_z', 'lmm_bw', 'cluster', 'ols')}
    est, ses, iccs = [], [], []
    for _ in range(reps):
        o, b, se, ih = analyse(*gen_trial(K, mbar, icc, delta, rng))
        for k in P:
            P[k] += o[k] < 0.05
        est.append(b); ses.append(se); iccs.append(ih)
    return {k: v / reps for k, v in P.items()} | dict(bias=float(np.mean(est) - delta), emp_se=float(np.std(est)),
                                                       mean_se=float(np.mean(ses)), icc_hat=float(np.mean(iccs)))


if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    t0 = time.time(); res = {}
    Ks = [12, 16, 20, 24, 28, 32, 40]
    jobs = []
    for icc in (0.02, 0.05, 0.10, 0.15):
        for d in (0.0, 1.0):
            for K in Ks:
                jobs.append((K, 45, icc, d))
    for icc in (0.05, 0.10):
        for d in (0.75, 1.5):
            for K in Ks:
                jobs.append((K, 45, icc, d))
    for mb in (30, 60):
        for K in Ks:
            jobs.append((K, mb, 0.05, 1.0))
    from multiprocessing import Pool
    args = [(K, mb, icc, d, reps, 1000 * i + 17) for i, (K, mb, icc, d) in enumerate(jobs)]
    with Pool(2) as p:
        out = p.starmap(cell, args)
    for (K, mb, icc, d), o in zip(jobs, out):
        res[f'K{K}_m{mb}_icc{icc}_d{d}'] = o
    json.dump(res, open('./s3.json', 'w'))
    print('done', time.time() - t0)
