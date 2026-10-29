"""Synthetic front-kick kinematics, camera projection and pose-estimation error model.

Used only to characterise the movement-quality score (MQS) and the frontal-plane
projection angle (FPPA) flag before the trial; no human data are involved.
"""
import numpy as np

FPS = 30
JOINTS = ['sh_L', 'sh_R', 'el_L', 'el_R', 'wr_L', 'wr_R', 'hip_L', 'hip_R', 'kn_L', 'kn_R', 'an_L', 'an_R', 'head']
J = {n: i for i, n in enumerate(JOINTS)}
FAULT_W = np.array([0.30, 0.25, 0.15, 0.20, 0.10])  # hip height, knee extension, trunk lean, valgus, timing
FAULT_NAMES = ['reduced hip flexion', 'incomplete knee extension', 'trunk lean-back', 'support-knee valgus',
               'chamber-timing delay']


def bump(s, a, b, p=2.0):
    """Smooth 0->1->0 profile on [a, b]."""
    x = np.clip((s - a) / (b - a), 0, 1)
    return np.sin(np.pi * x) ** p


def true_quality(sev):
    return 100.0 * (1.0 - float(np.dot(FAULT_W, sev)))


def kick_3d(sev, height=1.45, n_frames=33, sway=0.0, rng=None, tempo=1.0):
    """Return (T, 13, 3) world coordinates for a right front kick with fault severities sev[0..4]."""
    rng = rng or np.random.default_rng()
    T = max(18, int(round(n_frames * tempo)))
    s = np.linspace(0, 1, T)
    # timing fault: chamber is delayed and the strike phase compressed
    d = 0.18 * sev[4]
    warp = np.where(s < 0.5, s * (0.5 - 0) / 0.5, s)
    s_eff = np.clip(np.where(s < 0.35 + d, s * 0.35 / (0.35 + d), 0.35 + (s - 0.35 - d) * 0.65 / (0.65 - d)), 0, 1)
    del warp
    H = height
    thigh, shank, torso = 0.245 * H, 0.246 * H, 0.30 * H
    hw, sw = 0.055 * H, 0.10 * H
    ua, fa, head = 0.186 * H, 0.146 * H, 0.13 * H
    jit = lambda a: a + sway * rng.normal(0, 1)
    hip_peak = np.deg2rad(jit(95 - 40 * sev[0]))
    knee_min = np.deg2rad(jit(10 + 50 * sev[1]))
    lean_pk = np.deg2rad(jit(10 + 25 * sev[2]))
    valg_pk = np.deg2rad(max(0.0, jit(2 + 16 * sev[3])))
    sup_flex = np.deg2rad(jit(18))
    hflex = hip_peak * bump(s_eff, 0.08, 0.92, 1.2)
    # knee: flexed in chamber (~110 deg), extends at strike (s~0.5), re-chambers, then straightens
    chamber = np.deg2rad(110)
    k = np.interp(s_eff, [0, 0.25, 0.38, 0.5, 0.58, 0.72, 0.9, 1.0],
                  [0.05, chamber, chamber * 0.7, knee_min, knee_min, chamber, 0.3, 0.05])
    lean = lean_pk * bump(s_eff, 0.15, 0.85, 1.5)
    valg = valg_pk * bump(s_eff, 0.2, 0.8, 1.0)
    sflex = sup_flex * (0.6 + 0.4 * bump(s_eff, 0.1, 0.9, 1.0))
    P = np.zeros((T, 13, 3))
    # support (left) leg: ankle fixed on ground
    anL = np.array([-hw, 0.0, 0.0])
    hipH = (thigh + shank) * np.cos(sflex / 2)
    knL = np.stack([-hw + thigh * np.sin(valg), shank * np.cos(sflex / 2), shank * np.sin(sflex / 2)], -1)
    hipL = np.stack([np.full(T, -hw), hipH, np.zeros(T)], -1)
    hipR = np.stack([np.full(T, hw), hipH, np.zeros(T)], -1)
    thighv = np.stack([np.zeros(T), -np.cos(hflex), np.sin(hflex)], -1) * thigh
    knR = hipR + thighv
    a2 = hflex - k
    shankv = np.stack([np.zeros(T), -np.cos(a2), np.sin(a2)], -1) * shank
    anR = knR + shankv
    hipM = (hipL + hipR) / 2
    trunk = np.stack([np.zeros(T), np.cos(lean), -np.sin(lean)], -1) * torso
    shM = hipM + trunk
    shL = shM + np.array([-sw, 0, 0]); shR = shM + np.array([sw, 0, 0])
    hd = shM + trunk / torso * head
    # guard: upper arm down/forward, forearm up toward chin
    uav = np.array([0, -np.cos(0.5), np.sin(0.5)]) * ua
    fav = np.array([0, np.cos(0.3), np.sin(0.3)]) * fa
    elL = shL + uav + np.array([0.02, 0, 0]); elR = shR + uav - np.array([0.02, 0, 0])
    wrL = elL + fav; wrR = elR + fav
    for n, v in [('sh_L', shL), ('sh_R', shR), ('el_L', elL), ('el_R', elR), ('wr_L', wrL), ('wr_R', wrR),
                 ('hip_L', hipL), ('hip_R', hipR), ('kn_L', knL), ('kn_R', knR), ('an_L', np.broadcast_to(anL, (T, 3))),
                 ('an_R', anR), ('head', hd)]:
        P[:, J[n]] = v
    return P


def project(P, yaw_deg, dist=3.5, cam_h=1.0, f=900.0):
    """Perspective projection of (T,13,3) world points for a camera at yaw (deg) facing the child."""
    y = np.deg2rad(yaw_deg)
    R = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Q = P @ R.T
    depth = dist - Q[..., 2]
    u = f * Q[..., 0] / depth
    v = f * (Q[..., 1] - cam_h) / depth
    return np.stack([u, v], -1), Q


def occlusion(Q):
    """Self-occlusion proxy in [0,1]: joints behind the body plane relative to the camera."""
    zc = Q[..., 2] - np.median(Q[:, [J['hip_L'], J['hip_R'], J['sh_L'], J['sh_R']], 2], axis=1, keepdims=True)
    return np.clip(-zc / 0.12, 0, 1)


def observe(P, yaw, rng, sigma=0.03, p_drop=0.04):
    """Simulated 2D pose-estimator output: noisy keypoints (pixels) and confidences."""
    uv, Q = project(P, yaw)
    occ = occlusion(Q)
    torso_px = np.median(np.linalg.norm((uv[:, J['sh_L']] + uv[:, J['sh_R']]) / 2 - (uv[:, J['hip_L']] + uv[:, J['hip_R']]) / 2, axis=-1))
    sd = sigma * torso_px * (1 + 2.5 * occ)
    noisy = uv + rng.normal(0, 1, uv.shape) * sd[..., None]
    conf = np.clip(0.95 - 0.7 * occ + rng.normal(0, 0.05, occ.shape), 0.02, 1.0)
    drop = rng.random(occ.shape) < p_drop
    noisy[drop] += rng.normal(0, 0.35 * torso_px, (drop.sum(), 2))
    conf[drop] = rng.uniform(0.02, 0.2, drop.sum())
    return noisy, conf


# ---------------------------------------------------------------- feature extraction
def smooth(X, C, win=5):
    """Confidence-weighted moving average along time (keypoints, before angle computation)."""
    T = X.shape[0]
    out = np.empty_like(X)
    h = win // 2
    for t in range(T):
        a, b = max(0, t - h), min(T, t + h + 1)
        w = C[a:b][..., None] + 1e-6
        out[t] = (X[a:b] * w).sum(0) / w.sum(0)
    return out


def normalise(X):
    hipM = (X[:, J['hip_L']] + X[:, J['hip_R']]) / 2
    shM = (X[:, J['sh_L']] + X[:, J['sh_R']]) / 2
    L = np.median(np.linalg.norm(shM - hipM, axis=-1))
    return (X - hipM[:, None, :]) / L, L


def ang(a, b, c):
    """Interior angle at b (degrees) for 2D point arrays."""
    u, v = a - b, c - b
    cs = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cs, -1, 1)))


def features(X, C):
    """Per-frame angle vector A (T,5) and matching confidences W (T,5)."""
    sh = (X[:, J['sh_L']] + X[:, J['sh_R']]) / 2
    hip = (X[:, J['hip_L']] + X[:, J['hip_R']]) / 2
    up = hip + np.array([0, 1.0])
    A = np.stack([
        ang(sh, X[:, J['hip_R']], X[:, J['kn_R']]),        # kicking hip angle (trunk-thigh)
        ang(X[:, J['hip_R']], X[:, J['kn_R']], X[:, J['an_R']]),  # kicking knee angle
        ang(X[:, J['hip_L']], X[:, J['kn_L']], X[:, J['an_L']]),  # support knee angle
        ang(sh, hip, up),                                   # trunk inclination
        ang(X[:, J['sh_R']], X[:, J['el_R']], X[:, J['wr_R']]),   # guard elbow
    ], -1)
    c = lambda *n: np.prod([C[:, J[k]] for k in n], axis=0) ** (1 / len(n))
    W = np.stack([c('sh_L', 'sh_R', 'hip_R', 'kn_R'), c('hip_R', 'kn_R', 'an_R'), c('hip_L', 'kn_L', 'an_L'),
                  c('sh_L', 'sh_R', 'hip_L', 'hip_R'), c('sh_R', 'el_R', 'wr_R')], -1)
    return A, W


def fppa(X):
    """Signed 2D frontal-plane projection angle at the support knee (positive = medial collapse), degrees."""
    h, k, a = X[:, J['hip_L']], X[:, J['kn_L']], X[:, J['an_L']]
    th = ang(h, k, a)
    # medial direction for the left leg is +u (towards the body midline) in the image
    side = np.sign((k[:, 0] - (h[:, 0] + a[:, 0]) / 2) + 1e-12)
    return side * (180 - th)


# ---------------------------------------------------------------- alignment
def dtw(A, W, R, band=0.25, weighted=True):
    """Confidence-weighted DTW between observed angles A (T,d) and reference R (S,d).

    Returns accumulated mean cost and the warping path."""
    T, S = len(A), len(R)
    w = W if weighted else np.ones_like(W)
    diff = np.abs(A[:, None, :] - R[None, :, :])  # T,S,d
    cost = (diff * w[:, None, :]).sum(-1) / (w.sum(-1)[:, None] + 1e-9)
    D = np.full((T + 1, S + 1), np.inf); D[0, 0] = 0
    bw = max(int(band * max(T, S)), abs(T - S) + 1)
    for i in range(1, T + 1):
        j0 = max(1, int(i * S / T) - bw); j1 = min(S, int(i * S / T) + bw)
        for j in range(j0, j1 + 1):
            D[i, j] = cost[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    # backtrack
    i, j, path = T, S, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        k = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
        i, j = (i - 1, j - 1) if k == 0 else ((i - 1, j) if k == 1 else (i, j - 1))
    path = path[::-1]
    return D[T, S] / len(path), np.array(path)


def timing_dev(path, T, S):
    """Mean absolute deviation of the warping path from the diagonal, in normalised time."""
    p = np.asarray(path, float)
    return float(np.mean(np.abs(p[:, 0] / (T - 1) - p[:, 1] / (S - 1))))


KEY_REF = (10, 17)  # chamber and strike frames of the 33-frame reference


def key_error(A, W, R, path, key_ref=KEY_REF, half=1):
    """Phase-anchored angle error at the reference key events, mapped to observed frames by the path."""
    path = np.asarray(path)
    errs = []
    for kr in key_ref:
        obs = path[path[:, 1] == kr, 0]
        t = int(np.round(obs.mean())) if len(obs) else int(round(kr * (len(A) - 1) / (len(R) - 1)))
        a, b = max(0, t - half), min(len(A), t + half + 1)
        ra, rb = max(0, kr - half), min(len(R), kr + half + 1)
        oa = (A[a:b] * W[a:b]).sum(0) / (W[a:b].sum(0) + 1e-9)
        wa = W[a:b].mean(0)
        e = (np.abs(oa - R[ra:rb].mean(0)) * wa).sum() / (wa.sum() + 1e-9)
        errs.append(e)
    return float(np.mean(errs))
