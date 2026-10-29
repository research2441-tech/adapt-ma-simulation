"""Adds the delta = 1.5 cells for ICC 0.02 and 0.15 to s3.json (run after trialsim.py)."""
import json
from multiprocessing import Pool
import trialsim as T

if __name__ == '__main__':
    S = json.load(open('s3.json'))
    jobs = [(K, 45, icc, 1.5) for icc in (0.02, 0.15) for K in [12, 16, 20, 24, 28, 32, 40]]
    args = [(K, mb, icc, d, 4000, 500000 + 17 * i) for i, (K, mb, icc, d) in enumerate(jobs)]
    with Pool(2) as p:
        out = p.starmap(T.cell, args)
    for (K, mb, icc, d), o in zip(jobs, out):
        S[f'K{K}_m{mb}_icc{icc}_d{d}'] = o
    json.dump(S, open('s3.json', 'w'))
