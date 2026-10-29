"""Experiment S1: movement-quality score (MQS) and FPPA flag on synthetic front kicks."""
import json, sys, time
import numpy as np
from multiprocessing import Pool
from scipy.stats import spearmanr, rankdata
from kinematics import *

BANK_YAWS = np.arange(-60, 61, 15)
REF_H = 1.45


FPPA_REF = {}


def make_bank():
    bank = {}
    P = kick_3d(np.zeros(5), height=REF_H, n_frames=33)
    for y in BANK_YAWS:
        uv, Q = project(P, y)
        X, _ = normalise(uv)
        A, W = features(X, np.ones(X.shape[:2]))
        bank[int(y)] = A
        f = fppa(uv)[int(0.25 * len(uv)):int(0.75 * len(uv))]
        FPPA_REF[int(y)] = float(np.median(np.sort(f)[-5:]))
    return bank


BANK = make_bank()


def score(X, C, variant):
    """Return (angle error, timing deviation) for one observed performance under a scoring variant."""
    weighted = variant != 'no_conf'
    Cw = C if weighted else np.ones_like(C)
    Xs = smooth(X, Cw)
    Xn, _ = normalise(Xs)
    A, W = features(Xn, Cw)
    yaws = [45] if variant == 'single_view' else list(BANK)
    best = (np.inf, 0.0, 0.0)
    for y in yaws:
        R = BANK[y]
        if variant == 'linear_time':
            idx = np.linspace(0, len(R) - 1, len(A)).round().astype(int)
            e = float(((np.abs(A - R[idx]) * W).sum(-1) / (W.sum(-1) + 1e-9)).mean())
            path = np.stack([np.arange(len(A)), idx], -1)
            cand = (e, 0.0, key_error(A, W, R, path))
        else:
            e, path = dtw(A, W, R, band=0.1, weighted=weighted)
            cand = (e, timing_dev(path, len(A), len(R)), key_error(A, W, R, path))
        if cand[0] < best[0]:
            best = cand + (y,)
    best = best[:3] + (best[3] if len(best) > 3 else 0,)
    if variant == 'no_timing':
        best = (best[0], 0.0, best[2], best[3])
    if variant == 'no_keyphase':
        best = (best[0], best[1], best[0], best[3])
    return best


VARIANTS = ['full', 'no_conf', 'single_view', 'linear_time', 'no_timing', 'no_keyphase']


def simulate_one(args):
    seed, sigma, p_drop, yaw_fixed, variants = args
    rng = np.random.default_rng(seed)
    sev = np.clip(rng.beta(0.9, 2.2, 5), 0, 1)
    H = rng.uniform(1.28, 1.70)
    yaw = rng.choice([-1, 1]) * rng.uniform(25, 55) if yaw_fixed is None else yaw_fixed
    out = {'sev': sev.tolist(), 'Q': true_quality(sev), 'yaw': yaw, 'valgus_deg': 2 + 16 * sev[3]}
    for rep in (0, 1):
        tempo = rng.uniform(0.85, 1.25)
        P = kick_3d(sev, height=H, sway=2.0, rng=rng, tempo=tempo)
        X, C = observe(P, yaw + (rng.uniform(-5, 5) if rep else 0), rng, sigma=sigma, p_drop=p_drop)
        r = {v: score(X, C, v) for v in variants}
        Xs = smooth(X, C)
        T = len(Xs)
        f = fppa(Xs)[int(0.25 * T):int(0.75 * T)]
        r['fppa'] = float(np.median(np.sort(f)[-5:]))
        r['fppa_vr'] = r['fppa'] - FPPA_REF[int(r['full'][3])] if 'full' in r else r['fppa']
        out['rec%d' % rep] = r
    return out


def run(n, sigma=0.03, p_drop=0.04, yaw_fixed=None, variants=VARIANTS, seed0=0):
    args = [(seed0 + i, sigma, p_drop, yaw_fixed, variants) for i in range(n)]
    with Pool(8) as p:
        return p.map(simulate_one, args, chunksize=8)


if __name__ == '__main__':
    t0 = time.time()
    res = {}
    res['main'] = run(1600, seed0=10_000)
    print('main done', time.time() - t0, flush=True)
    sweeps = {}
    for s in [0.01, 0.02, 0.03, 0.05, 0.08]:
        sweeps['sigma_%g' % s] = run(400, sigma=s, variants=['full', 'no_conf', 'single_view'], seed0=int(200_000 + s * 1e5))
        print('sigma', s, time.time() - t0, flush=True)
    for pdp in [0.0, 0.1, 0.2]:
        sweeps['drop_%g' % pdp] = run(400, p_drop=pdp, variants=['full', 'no_conf', 'single_view'], seed0=int(300_000 + pdp * 1e4))
        print('drop', pdp, time.time() - t0, flush=True)
    for y in [0, 10, 20, 30, 40, 50, 60]:
        sweeps['yaw_%d' % y] = run(400, yaw_fixed=float(y), variants=['full', 'single_view'], seed0=400_000 + y * 100)
        print('yaw', y, time.time() - t0, flush=True)
    res['sweeps'] = sweeps
    json.dump(res, open('./s1_raw.json', 'w'))
    print('total', time.time() - t0)
