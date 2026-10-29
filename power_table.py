"""Analytic power approximation (design effect x ANCOVA factor) next to the simulated power from trialsim.py."""
import json
import numpy as np
from scipy import stats

S3 = json.load(open('s3.json'))


def analytic(K, icc, d, m=45, attr=0.15, cv=0.4, sd=6.0, ri=0.7, rc=0.8):
    me = m * (1 - attr)
    de = 1 + ((cv ** 2 + 1) * me - 1) * icc
    r = (me * icc * rc + (1 - icc) * ri) / (1 + (me - 1) * icc)
    se = np.sqrt(4 * sd ** 2 * de * (1 - r ** 2) / (K * me))
    t = stats.t.ppf(0.975, K - 2)
    return 1 - stats.nct.cdf(t, K - 2, d / se)


out = {}
for icc in (0.02, 0.05, 0.10, 0.15):
    for K in (20, 24, 28, 32):
        sim = S3[f'K{K}_m45_icc{icc}_d1.5']['lmm_bw']
        out[f'{icc}_{K}'] = dict(analytic=float(analytic(K, icc, 1.5)), sim=sim)
        print(icc, K, round(sim, 3), round(out[f'{icc}_{K}']['analytic'], 3))
json.dump(out, open('s3_table.json', 'w'))
