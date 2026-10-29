from diagram import Diagram

OUT = './'


def fig_trial():
    d = Diagram(7.0, 3.05)
    W = 2.7
    d.box('elig', 3.5, 2.78, 4.6, 0.36, 'Primary and junior-secondary schools assessed for eligibility, Henan Province, China\n'
          'Children aged 8–15 years; parental consent and child assent', size=6.8)
    d.box('base', 3.5, 2.18, 4.6, 0.36, 'Baseline assessment T0 (weeks −2 to 0): 20-m shuttle run, accelerometry, Flanker,\n'
          '2-back, SDQ, CYRM-R, BMI, screen time, motor competence', size=6.8)
    d.box('rand', 3.5, 1.62, 4.6, 0.3, r'Stratified cluster randomisation 1:1 by school ($K=28$; strata: urban/rural, enrolment size)',
          size=6.8, bold=False)
    d.box('int', 1.65, 1.02, W, 0.5, 'Intervention, 14 schools: pose-guided\nadaptive martial arts, 3 × 45 min/week,\n'
          '16 weeks, smartphone MQS and FPPA flags', size=6.4)
    d.box('con', 5.35, 1.02, W, 0.5, 'Active control, 14 schools: standard\nphysical education, 3 × 45 min/week,\n'
          '16 weeks, volume matched', size=6.4)
    d.box('t1', 1.65, 0.45, W, 0.3, 'Post-intervention T1 (week 17); follow-up T2 (week 28)', size=6.6)
    d.box('t1c', 5.35, 0.45, W, 0.3, 'Post-intervention T1 (week 17); follow-up T2 (week 28)', size=6.6)
    d.arrow(d.anchor('elig', 'b'), d.anchor('base', 't'))
    d.arrow(d.anchor('base', 'b'), d.anchor('rand', 't'))
    d.poly([d.anchor('rand', 'b', 0.25), (d.anchor('rand', 'b', 0.25)[0], 1.42), (1.65, 1.42), d.anchor('int', 't')])
    d.poly([d.anchor('rand', 'b', 0.75), (d.anchor('rand', 'b', 0.75)[0], 1.42), (5.35, 1.42), d.anchor('con', 't')])
    d.arrow(d.anchor('int', 'b'), d.anchor('t1', 't'))
    d.arrow(d.anchor('con', 'b'), d.anchor('t1c', 't'))
    d.note(3.5, 0.98, 'Injury\nsurveillance,\nboth arms', size=6.2, style='italic')
    d.note(3.5, 0.12, 'Primary estimand: between-arm difference in estimated VO$_{2\\mathrm{peak}}$ at T1, treatment-policy strategy',
           size=6.4)
    d.save(OUT + 'fig1_trial.png')


def fig_arch():
    d = Diagram(7.0, 3.35)
    w, h = 1.0, 0.5
    y1, y2, y3 = 2.78, 1.62, 0.5
    xs = [0.6, 1.78, 2.96, 4.14, 5.32, 6.45]
    d.box('child', xs[0], y1, w, h, 'Child performs\ntechnique', size=6.6)
    d.box('cam', xs[1], y1, w, h, 'Smartphone video\n30 fps, yaw 30–50°', size=6.6)
    d.box('pose', xs[2], y1, w, h, '2D pose estimator\nkeypoints + conf.', size=6.6)
    d.box('norm', xs[3], y1, w, h, 'Conf.-weighted\nsmoothing, scaling', size=6.6)
    d.box('dtw', xs[4], y1, w, h, 'View-matched bank\nweighted DTW', size=6.6)
    d.box('mqs', xs[5], y1, 0.95, h, 'MQS: trajectory,\nkey phase, timing', size=6.6)
    d.box('rpe', xs[1], y2, w, h, 'Session RPE,\nattendance', size=6.6)
    d.box('load', xs[2], y2, w, h, 'Acute/chronic\nload (EWMA)', size=6.6)
    d.box('fppa', xs[3], y2, w, h, 'View-referenced\nFPPA', size=6.6)
    d.box('flag', xs[4], y2, w, h, 'Valgus flag\nrate', size=6.6)
    d.box('ekf', xs[5], y2, 0.95, h, 'EKF skill\nstate update', size=6.6)
    d.box('ctrl', 4.2, y3, 4.6, 0.56, 'Session controller: level choice targeting success probability $p^{\\ast}$,\n'
          'intensity guard on acute:chronic ratio band, lower-limb cap after valgus flags', size=6.6)
    d.box('plan', 0.95, y3, 1.5, 0.56, 'Next-session plan\n(levels, intensity)', size=6.6)
    for a, b in (('child', 'cam'), ('cam', 'pose'), ('pose', 'norm'), ('norm', 'dtw'), ('dtw', 'mqs')):
        d.arrow(d.anchor(a, 'r'), d.anchor(b, 'l'))
    d.arrow(d.anchor('norm', 'b'), d.anchor('fppa', 't'))
    d.poly([d.anchor('dtw', 'b', 0.3), (d.anchor('dtw', 'b', 0.3)[0], 2.2), (d.anchor('fppa', 't', 0.8)[0], 2.2),
            d.anchor('fppa', 't', 0.8)], label=r'$\hat{\psi}$', lseg=0, loff=(0.1, 0.0))
    d.arrow(d.anchor('fppa', 'r'), d.anchor('flag', 'l'))
    d.arrow(d.anchor('mqs', 'b'), d.anchor('ekf', 't'), label=r'$s_{c,a}$', loff=(0.14, 0.0))
    d.arrow(d.anchor('rpe', 'r'), d.anchor('load', 'l'))
    d.arrow(d.anchor('load', 'b'), (xs[2], y3 + 0.28))
    d.arrow(d.anchor('flag', 'b'), (xs[4], y3 + 0.28))
    d.arrow(d.anchor('ekf', 'b'), (xs[5], y3 + 0.28), label=r'$\hat{\theta}_{c,f}$', loff=(0.16, 0.0))
    d.arrow(d.anchor('ctrl', 'l'), d.anchor('plan', 'r'))
    d.poly([d.anchor('plan', 't', 0.1), (d.anchor('plan', 't', 0.1)[0], y1 - h / 2)], dashed=True,
           label='next\nsession', lseg=0, loff=(0.0, -0.1))
    d.poly([d.anchor('child', 'b', 0.8), (d.anchor('child', 'b', 0.8)[0], y2), d.anchor('rpe', 'l')],
           label='self-report', lseg=0, loff=(0.3, 0.0))
    d.save(OUT + 'fig2_arch.png')


if __name__ == '__main__':
    fig_trial(); fig_arch(); print('ok')
