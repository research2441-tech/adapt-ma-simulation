"""Experiment S2: adaptive difficulty and load controller on a synthetic cohort.

Latent-skill learning follows a challenge-point response: per-session gain peaks when the
success probability of the practised level is intermediate. Observations are noisy MQS
values whose error SD is taken from experiment S1.
"""
import json, sys
import numpy as np

N_FAM = 6
LEVELS = np.linspace(-2.0, 2.5, 6)          # difficulty b_l of levels 1..6
LOWER_LIMB = np.array([1, 1, 0, 1, 0, 0])    # families whose difficulty raises valgus exposure
WEEKS, SPW = 16, 3
N_SESS = WEEKS * SPW
ATT = 8                                       # attempts per family per session
P_TARGET = 0.70


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def gain(p, width=0.15):
    return np.exp(-(p - P_TARGET) ** 2 / (2 * width ** 2))


def cohort(n, rng):
    age = rng.integers(8, 16, n)
    theta0 = (-1.4 + 0.12 * (age - 8))[:, None] + rng.normal(0, 0.6, (n, N_FAM))
    eta = np.exp(rng.normal(np.log(0.055), 0.4, n))
    valg = rng.normal(-3.2, 0.9, n)               # latent valgus tendency on the logit scale
    rpe_bias = rng.normal(0, 0.6, n)
    # attendance: independent misses plus illness blocks of one week
    att = rng.random((n, N_SESS)) > 0.07
    for c in range(n):
        for _ in range(rng.poisson(0.8)):
            s0 = rng.integers(0, N_SESS - 3)
            att[c, s0:s0 + 3] = False
    return dict(age=age, theta0=theta0, eta=eta, valg=valg, rpe_bias=rpe_bias, att=att)


def fixed_intensity(s):
    w = s // SPW
    return [0.45, 0.60, 0.75, 0.85][min(3, w // 4)]


def run_policy(C, policy, sigma_m, rng, p_target=P_TARGET, acwr_band=(0.8, 1.3), flag_cap=0.25, dec=1):
    n = len(C['age'])
    theta = C['theta0'].copy()
    mu = np.full((n, N_FAM), -1.0); P = np.full((n, N_FAM), 1.0)   # EKF prior on skill
    q = 0.02                                                          # process noise per session
    base = 45 * (3 + 5 * 0.45)                                        # habitual PE load before week 1
    acute = np.full(n, base) + 45 * C['rpe_bias']; chronic = acute.copy(); seen = np.zeros(n)
    la, lc = 2 / (SPW + 1), 2 / (4 * SPW + 1)
    rec = dict(inband=np.zeros(n), attempts=np.zeros(n), spikes=np.zeros(n), risky=np.zeros(n),
               load=np.zeros(n), flags=np.zeros(n))
    flag_rate = np.zeros(n)
    rec['ratio_t'], rec['theta_t'], rec['inband_t'], rec['u_t'], rec['ratio_p95'] = [], [], [], [], []
    for s in range(N_SESS):
        present = C['att'][:, s]
        # ---- choose level per family
        if policy == 'fixed':
            lvl = np.full((n, N_FAM), min(5, (s // SPW) // 3))
        elif policy == 'oracle':
            lvl = np.argmin(np.abs(sig(theta[..., None] - LEVELS) - p_target), -1)
        else:
            lvl = np.argmin(np.abs(sig(mu[..., None] - LEVELS) - p_target), -1)
            if policy == 'full':
                cap = (flag_rate > flag_cap)[:, None] & (LOWER_LIMB[None, :] == 1)
                lvl = np.where(cap, np.maximum(lvl - dec, 0), lvl)
        b = LEVELS[lvl]
        # ---- choose intensity (fraction of capacity)
        u_plan = np.full(n, fixed_intensity(s))
        if policy == 'full':
            # load guard: keep the projected acute:chronic ratio inside the band
            lo, hi = acwr_band
            u_max = ((hi * chronic - (1 - la) * acute) / (la * 45) - 3 - C['rpe_bias']) / 5
            u_min = ((lo * chronic - (1 - la) * acute) / (la * 45) - 3 - C['rpe_bias']) / 5
            u_plan = np.clip(np.clip(u_plan, u_min, u_max), 0.3, 0.95)
        # ---- execute session for present children
        p = sig(theta - b)
        k = ATT * present[:, None]
        rec['inband'] += ((p > 0.55) & (p < 0.85)) .sum(1) * ATT * present
        rec['attempts'] += N_FAM * ATT * present
        fatigue = u_plan
        p_flag = sig(C['valg'][:, None] + 0.35 * lvl + 1.2 * fatigue[:, None]) * LOWER_LIMB[None, :]
        flags = rng.binomial(ATT, p_flag) * present[:, None]
        rec['flags'] += flags.sum(1)
        rec['risky'] += ((p_flag > 0.15) * ATT * present[:, None]).sum(1)
        flag_rate = np.where(present, flags.sum(1) / (ATT * LOWER_LIMB.sum()), flag_rate)
        theta += present[:, None] * C['eta'][:, None] * gain(p) * (0.6 + 0.8 * fatigue[:, None])
        # observations (MQS on 0-100 scale) and EKF update
        if policy in ('adaptive', 'full'):
            for _ in range(2):   # two scored repetitions per family per session
                y = 100 * sig(theta - b) + rng.normal(0, sigma_m, theta.shape)
                Pm = P + q
                h = 100 * sig(mu - b); H = 100 * sig(mu - b) * (1 - sig(mu - b))
                K = Pm * H / (H ** 2 * Pm + sigma_m ** 2)
                mu = np.where(present[:, None], mu + K * (y - h), mu)
                P = np.where(present[:, None], (1 - K * H) * Pm, P + q)
        # ---- load bookkeeping (absent sessions carry zero load)
        rpe = np.clip(3 + 5 * u_plan + C['rpe_bias'], 1, 10)
        L = 45 * rpe * present
        acute = la * L + (1 - la) * acute
        chronic = lc * L + (1 - lc) * chronic
        seen += present
        ratio = acute / np.maximum(chronic, 1e-6)
        rec['spikes'] += (ratio > 1.3) & present
        rec['ratio_t'].append(float(np.mean(ratio[present]))); rec['ratio_p95'].append(float(np.percentile(ratio[present], 95)))
        rec['theta_t'].append(float(theta.mean())); rec['u_t'].append(float(u_plan.mean()))
        rec['inband_t'].append(float(100 * (((p > 0.55) & (p < 0.85)).mean(1)[present]).mean()))
        rec['load'] += L
    rec['gain'] = (theta - C['theta0']).mean(1)
    rec['inband_pct'] = 100 * rec['inband'] / np.maximum(rec['attempts'], 1)
    rec['track_rmse'] = float(np.sqrt(((mu - theta) ** 2).mean())) if policy in ('adaptive', 'full') else float('nan')
    return rec


def summarise(rec, rng, B=1000):
    out = {}
    for k in ('gain', 'inband_pct', 'spikes', 'risky', 'load', 'flags'):
        x = np.asarray(rec[k], float)
        bs = [rng.choice(x, len(x)).mean() for _ in range(B)]
        out[k] = [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    out['track_rmse'] = rec['track_rmse']
    for k in ('ratio_t', 'theta_t', 'inband_t', 'u_t', 'ratio_p95'):
        out[k] = rec[k]
    out['gain_children'] = np.asarray(rec['gain']).tolist()
    return out


if __name__ == '__main__':
    sigma_m = float(sys.argv[1])
    rng = np.random.default_rng(2026)
    C = cohort(1200, rng)
    res = {'sigma_m': sigma_m, 'main': {}}
    for pol in ('fixed', 'adaptive', 'nocap', 'full', 'oracle'):
        if pol == 'nocap':
            r = run_policy(C, 'full', sigma_m, np.random.default_rng(7), flag_cap=np.inf)
        else:
            r = run_policy(C, pol, sigma_m, np.random.default_rng(7))
        res['main'][pol] = summarise(r, np.random.default_rng(1))
    # sensitivity to measurement error and target success probability
    res['sens_sigma'] = {}
    for mult in (0.5, 1, 2, 4):
        r = run_policy(C, 'full', sigma_m * mult, np.random.default_rng(7))
        res['sens_sigma'][str(mult)] = [float(np.mean(r['gain'])), float(np.mean(r['inband_pct'])), r['track_rmse']]
    res['sens_target'] = {}
    for pt in (0.5, 0.6, 0.7, 0.8, 0.9):
        r = run_policy(C, 'full', sigma_m, np.random.default_rng(7), p_target=pt)
        res['sens_target'][str(pt)] = [float(np.mean(r['gain'])), float(np.mean(r['inband_pct'])), float(np.mean(r['risky']))]
    res['pareto'] = []
    for dec in (1, 2):
        for fc in (0.10, 0.15, 0.20, 0.25, 0.35, 0.50):
            r = run_policy(C, 'full', sigma_m, np.random.default_rng(7), flag_cap=fc, dec=dec)
            res['pareto'].append([dec, fc, float(np.mean(r['gain'])), float(np.mean(r['risky'])), float(np.mean(r['inband_pct']))])
    print(res['pareto'])
    # weekly trajectories for the figure: re-run tracking mean ratio per session
    json.dump(res, open('./s2.json', 'w'))
    for pol, v in res['main'].items():
        print(pol, {k: [round(a, 2) for a in v[k]] for k in ('gain', 'inband_pct', 'spikes', 'risky', 'load')}, v['track_rmse'])
    print(res['sens_sigma']); print(res['sens_target'])
