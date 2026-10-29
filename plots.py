"""Grayscale result figures (all values read from the simulation outputs)."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '.')

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Liberation Serif', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 7.5, 'axes.linewidth': 0.6, 'axes.labelsize': 7.5,
    'xtick.labelsize': 6.8, 'ytick.labelsize': 6.8, 'legend.fontsize': 6.5, 'lines.linewidth': 1.0,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'legend.frameon': True, 'legend.edgecolor': '0.6', 'legend.fancybox': False, 'legend.framealpha': 1.0,
    'savefig.dpi': 600, 'axes.grid': True, 'grid.color': '0.88', 'grid.linewidth': 0.4,
})
OUT = './'
COL = 3.45      # single column width in inches
WIDE = 7.0


def tag(ax, s):
    ax.set_xlabel(ax.get_xlabel() + '\n' + s)


def leg(ax, ncol=1, **kw):
    kw.setdefault('handlelength', 1.8); kw.setdefault('columnspacing', 0.9)
    kw.pop('bbox_to_anchor', None); kw.pop('borderaxespad', None)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=ncol, borderaxespad=0.0, **kw)


def fig_mqs_illustration():
    import kinematics as K
    from run_mqs import BANK, smooth, normalise, features, dtw
    rng = np.random.default_rng(42)
    sev = np.array([0.15, 0.7, 0.2, 0.1, 0.6])
    P = K.kick_3d(sev, height=1.40, sway=1.0, rng=rng, tempo=1.18)
    X, C = K.observe(P, 40.0, rng)
    Xn, _ = normalise(smooth(X, C)); A, W = features(Xn, C)
    R = BANK[45]
    e, path = dtw(A, W, R, band=0.1)
    fig, axs = plt.subplots(1, 2, figsize=(COL, 1.85), gridspec_kw=dict(width_ratios=[1.45, 1], wspace=0.42))
    ax = axs[0]
    tr = np.linspace(0, 1, len(R)); to = np.linspace(0, 1, len(A))
    ax.plot(tr, R[:, 1], color='black', lw=1.1, label='Reference')
    ax.plot(to, A[:, 1], color='0.45', lw=1.0, ls='--', marker='o', ms=1.8, label='Observed')
    for i, j in path[::3]:
        ax.plot([to[i], tr[j]], [A[i, 1], R[j, 1]], color='0.7', lw=0.4, zorder=0)
    for kr, nm in zip(K.KEY_REF, ('chamber', 'strike')):
        ax.axvline(tr[kr], color='0.3', lw=0.5, ls=':')
        ax.text(tr[kr] + 0.015, 50, nm, fontsize=6, rotation=90, va='bottom', ha='left')
    ax.set_xlabel('Normalised time'); ax.set_ylabel('Kicking-knee angle (deg)')
    ax.set_ylim(40, 200); ax.set_xlim(0, 1)
    leg(ax, ncol=2, bbox_to_anchor=(0.5, 1.0), borderaxespad=0.3, handlelength=1.6)
    tag(ax, '(a)')
    ax = axs[1]
    ax.plot(path[:, 1] / (len(R) - 1), path[:, 0] / (len(A) - 1), color='black', lw=1.1, label='Warping path')
    ax.plot([0, 1], [0, 1], color='0.55', lw=0.8, ls='--', label='Diagonal')
    ax.set_xlabel('Reference time'); ax.set_ylabel('Observed time')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.0)
    leg(ax, handlelength=1.6)
    tag(ax, '(b)')
    fig.savefig(OUT + 'fig3_mqs_alignment.png', bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)


def fig_mqs_results():
    S1 = json.load(open(OUT + 's1.json')); neg = json.load(open(OUT + 's1_negyaw.json'))
    sw = S1['sweep']; sw.update(neg)
    fig, axs = plt.subplots(1, 3, figsize=(WIDE, 2.05), gridspec_kw=dict(wspace=0.36))
    ax = axs[0]
    q = np.array(S1['scatter']['Q']); s = np.array(S1['scatter']['full'])
    ax.scatter(q, s, s=3, c='0.35', lw=0, alpha=0.75, rasterized=True)
    ax.plot([30, 100], [30, 100], color='black', lw=0.8, ls='--', label='Identity')
    b1, b0 = np.polyfit(q, s, 1); xx = np.array([30, 100])
    ax.plot(xx, b0 + b1 * xx, color='black', lw=1.0, label='Least-squares fit')
    ax.set_xlim(30, 100); ax.set_ylim(30, 110)
    ax.set_xlabel('Simulated true quality $Q$'); ax.set_ylabel('Calibrated MQS')
    leg(ax, handlelength=1.8)
    tag(ax, '(a)')
    ax = axs[1]
    yaws = sorted(int(k.split('_')[1]) for k in sw if k.startswith('yaw_'))
    for v, ls, mk, lab in (('full', '-', 'o', 'View-matched bank'), ('single_view', '--', 's', r'Single template (45$^\circ$)')):
        ax.plot(yaws, [sw['yaw_%d' % y][v]['rho'] for y in yaws], color='black' if v == 'full' else '0.45', ls=ls,
                marker=mk, ms=3, label=lab)
    ax.axvspan(25, 55, color='0.9', zorder=0, lw=0)
    ax.axvspan(-55, -25, color='0.9', zorder=0, lw=0)
    ax.set_xlabel('Camera yaw (deg)'); ax.set_ylabel(r"Spearman $\rho$ with $Q$")
    ax.set_xticks([-60, -40, -20, 0, 20, 40, 60]); ax.set_ylim(-0.2, 1.05)
    leg(ax, handlelength=2.0)
    tag(ax, '(b)')
    ax = axs[2]
    sig = [0.01, 0.02, 0.03, 0.05, 0.08]
    for v, ls, mk, c, lab in (('full', '-', 'o', 'black', 'Full MQS'), ('no_conf', '-.', '^', '0.35', 'No confidence weights'),
                              ('single_view', '--', 's', '0.55', 'Single template')):
        ax.plot(sig, [sw['sigma_%g' % x][v]['rho'] for x in sig], color=c, ls=ls, marker=mk, ms=3, label=lab)
    ax.set_xlabel(r'Keypoint noise $\sigma_{\mathrm{kp}}$ (torso lengths)'); ax.set_ylabel(r"Spearman $\rho$ with $Q$")
    ax.set_ylim(0.2, 1.05)
    leg(ax, handlelength=2.0)
    tag(ax, '(c)')
    fig.savefig(OUT + 'fig4_mqs_results.png', bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)


def fig_fppa():
    S1 = json.load(open(OUT + 's1.json')); neg = json.load(open(OUT + 's1_negyaw.json'))
    sw = S1['sweep']; sw.update(neg)
    fig, axs = plt.subplots(1, 2, figsize=(COL, 1.8), gridspec_kw=dict(wspace=0.42))
    ax = axs[0]
    for key, c, ls, lab in (('fppa_vr', 'black', '-', 'View-referenced'), ('fppa', '0.5', '--', 'Raw 2D FPPA')):
        fr, tr = S1['fppa'][key]['roc']
        ax.plot(fr, tr, color=c, ls=ls, lw=1.0, label=f"{lab} ({S1['fppa'][key]['auc']:.2f})")
    ax.plot([0, 1], [0, 1], color='0.75', lw=0.6, ls=':')
    ax.set_xlabel('1 − specificity'); ax.set_ylabel('Sensitivity'); ax.set_xlim(0, 1); ax.set_ylim(0, 1.0)
    leg(ax, handlelength=1.6, title='AUC', title_fontsize=6.5)
    tag(ax, '(a)')
    ax = axs[1]
    yaws = sorted(int(k.split('_')[1]) for k in sw if k.startswith('yaw_'))
    ax.plot(yaws, [sw['yaw_%d' % y]['fppa_vr_auc'] for y in yaws], color='black', marker='o', ms=3, label='View-referenced')
    ax.plot(yaws, [sw['yaw_%d' % y]['fppa_auc'] for y in yaws], color='0.5', ls='--', marker='s', ms=3, label='Raw 2D FPPA')
    ax.axvspan(25, 55, color='0.9', zorder=0, lw=0)
    ax.set_xlabel('Camera yaw (deg)'); ax.set_ylabel('AUC at fixed yaw'); ax.set_ylim(0.6, 1.0)
    ax.set_xticks([-60, -30, 0, 30, 60])
    leg(ax, handlelength=1.8)
    tag(ax, '(b)')
    fig.savefig(OUT + 'fig5_fppa.png', bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)


def fig_controller():
    S2 = json.load(open(OUT + 's2.json'))
    M = S2['main']
    sty = {'fixed': ('0.55', '--', 's', 'Fixed curriculum'), 'adaptive': ('0.3', '-.', '^', 'Adaptive, skill only'),
           'full': ('black', '-', 'o', 'Full controller'), 'oracle': ('0.7', ':', None, 'Oracle')}
    fig, axs = plt.subplots(1, 3, figsize=(WIDE, 2.05), gridspec_kw=dict(wspace=0.36))
    wk = np.arange(48) / 3 + 1 / 3
    ax = axs[0]
    for pol in ('fixed', 'adaptive', 'full', 'oracle'):
        c, ls, mk, lab = sty[pol]
        y = np.convolve(M[pol]['inband_t'], np.ones(3) / 3, mode='valid')
        ax.plot(wk[1:-1], y, color=c, ls=ls, label=lab)
    ax.set_xlabel('Week'); ax.set_ylabel('Attempts in challenge band (%)'); ax.set_ylim(0, 115); ax.set_xlim(0, 16.5)
    leg(ax, handlelength=2.0, ncol=2, columnspacing=0.8)
    tag(ax, '(a)')
    ax = axs[1]
    for pol in ('fixed', 'full'):
        c, ls, mk, lab = sty[pol]
        ax.plot(wk, M[pol]['ratio_p95'], color=c, ls=ls, label=lab + ', 95th pct.')
    ax.axhline(1.3, color='0.3', lw=0.6, ls=':')
    ax.text(0.6, 1.315, 'upper bound 1.3', fontsize=6, va='bottom')
    ax.set_xlabel('Week'); ax.set_ylabel('Acute:chronic load ratio'); ax.set_xlim(0, 16.5); ax.set_ylim(0.9, 1.42)
    leg(ax, handlelength=2.0)
    tag(ax, '(b)')
    ax = axs[2]
    par = np.array(S2['pareto'])
    for dec, c, mk, lab in ((1, 'black', 'o', 'Level decrement 1'), (2, '0.5', 's', 'Level decrement 2')):
        p = par[par[:, 0] == dec]
        ax.plot(p[:, 3], p[:, 2], color=c, marker=mk, ms=3, label=lab)
        if dec == 1:
            for row in p:
                if round(row[1], 2) in (0.10, 0.15, 0.20, 0.25):
                    ax.annotate(f'{row[1]:.2f}', (row[3], row[2]), textcoords='offset points', xytext=(-5, 3),
                                fontsize=5.5, ha='right', va='bottom')
    ax.set_xlabel('High-risk exposures per child'); ax.set_ylabel(r'Mean skill gain $\Delta\theta$')
    ax.set_ylim(2.08, 2.52); ax.set_xlim(290, 460)
    leg(ax, handlelength=1.8)
    tag(ax, '(c)')
    fig.savefig(OUT + 'fig6_controller.png', bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)


def fig_power():
    S3 = json.load(open(OUT + 's3.json'))
    Ks = [12, 16, 20, 24, 28, 32, 40]
    fig, axs = plt.subplots(1, 2, figsize=(WIDE, 2.05), gridspec_kw=dict(wspace=0.28))
    ax = axs[0]
    for icc, c, mk in ((0.02, 'black', 'o'), (0.05, '0.3', 's'), (0.10, '0.5', '^'), (0.15, '0.7', 'D')):
        ax.plot(Ks, [S3[f'K{K}_m45_icc{icc}_d1.5']['lmm_bw'] for K in Ks], color=c, marker=mk, ms=3,
                label=fr'$\rho_{{\mathrm{{ICC}}}}={icc:.2f}$')
    ax.axhline(0.9, color='0.3', ls=':', lw=0.7); ax.axhline(0.8, color='0.3', ls=':', lw=0.7)
    ax.axvline(28, color='black', lw=0.7, ls='--')
    ax.set_xlabel('Number of schools $K$'); ax.set_ylabel(r'Power ($\delta=1.5$ mL kg$^{-1}$ min$^{-1}$)')
    ax.set_ylim(0, 1.05); ax.set_xticks(Ks)
    leg(ax, handlelength=1.8)
    tag(ax, '(a)')
    ax = axs[1]
    meth = (('ols', 'Individual OLS', '0.75', '////'), ('lmm_z', 'LMM, normal ref.', '0.5', '....'),
            ('lmm_bw', 'LMM, $t_{K-2}$ ref.', '0.25', None), ('cluster', 'Cluster-level ANCOVA', 'white', 'xxxx'))
    x = np.arange(len(Ks)); w = 0.2
    for i, (k, lab, fc, h) in enumerate(meth):
        ax.bar(x + (i - 1.5) * w, [S3[f'K{K}_m45_icc0.1_d0.0'][k] for K in Ks], w, color=fc, edgecolor='black',
               lw=0.4, hatch=h, label=lab)
    ax.axhline(0.05, color='black', lw=0.7, ls='--')
    ax.set_xticks(x); ax.set_xticklabels(Ks); ax.set_xlabel('Number of schools $K$')
    ax.set_ylabel(r'Type I error ($\rho_{\mathrm{ICC}}=0.10$)'); ax.set_ylim(0, 0.36)
    leg(ax, ncol=2, handlelength=1.4, columnspacing=0.8)
    tag(ax, '(b)')
    fig.savefig(OUT + 'fig7_power.png', bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)


def fig_causal():
    S4 = json.load(open(OUT + 's4.json'))
    Ks = ['16', '24', '32']
    fig, axs = plt.subplots(1, 2, figsize=(COL, 1.9), gridspec_kw=dict(wspace=0.5))
    ax = axs[0]
    x = np.arange(3); w = 0.26
    for i, (k, lab, fc, h) in enumerate((('naive', 'Within-arm', '0.8', '////'), ('as_treated', 'As-treated', '0.5', None),
                                         ('iv', 'IV (2SLS)', 'black', None))):
        ax.bar(x + (i - 1) * w, [S4['dose'][K][k]['bias'] for K in Ks], w, color=fc, edgecolor='black', lw=0.4, hatch=h,
               label=lab)
    ax.axhline(0, color='black', lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(Ks); ax.set_xlabel('Schools $K$'); ax.set_ylabel(r'Bias (mL kg$^{-1}$ min$^{-1}$ per unit dose)')
    ax.set_ylim(-0.5, 5.2)
    leg(ax, handlelength=1.4)
    tag(ax, '(a)')
    ax = axs[1]
    for i, (k, lab, fc, h) in enumerate((('naive_noL', 'Naive, no $L$', '0.8', '////'),
                                         ('naive_adjL', 'Naive, $L$ adj.', '0.5', None),
                                         ('interventional', 'Interventional', 'black', None))):
        ax.bar(x + (i - 1) * w, [S4['med'][K][k]['cover'] for K in Ks], w, color=fc, edgecolor='black', lw=0.4, hatch=h,
               label=lab)
    ax.axhline(0.95, color='black', lw=0.7, ls='--')
    ax.set_xticks(x); ax.set_xticklabels(Ks); ax.set_xlabel('Schools $K$'); ax.set_ylabel('95% CI coverage (IIE)')
    ax.set_ylim(0, 1.05); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    leg(ax, handlelength=1.4, ncol=1, fontsize=6)
    tag(ax, '(b)')
    fig.savefig(OUT + 'fig8_causal.png', bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)


if __name__ == '__main__':
    which = sys.argv[1:] or ['mqs', 'mqsres', 'fppa', 'ctrl', 'power', 'causal']
    fn = dict(mqs=fig_mqs_illustration, mqsres=fig_mqs_results, fppa=fig_fppa, ctrl=fig_controller, power=fig_power,
              causal=fig_causal)
    for w in which:
        fn[w](); print('ok', w)
