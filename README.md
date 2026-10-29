# ADAPT-MA pre-enrolment simulation studies

Code for the simulation studies in *Simulation-Calibrated Protocol of a Cluster Randomised Trial Testing
Pose-Estimation-Guided Adaptive Martial Arts Training for Fitness, Executive Function and Wellbeing in Schoolchildren*.
No participant data are included or needed; every input is synthetic.

| File | Purpose |
|---|---|
| `kinematics.py` | 3D front-kick model with five graded faults, camera projection, pose-estimator error model, angle features, confidence-weighted DTW, key-phase error, FPPA |
| `run_mqs.py` | Study S1: movement-quality score variants and valgus flag on 1600 performances, plus robustness sweeps |
| `analyse_s1.py` | S1 calibration (600), held-out evaluation (1000), ablation deltas, ICC(A,1), MDC95, ROC |
| `negyaw.py` | S1 sweep over negative camera yaw (kicking-leg side) |
| `controller.py` | Study S2: extended-Kalman skill tracking, challenge-targeted levels, load-ratio guard, safety cap |
| `trialsim.py` | Study S3: cluster-trial simulation with profile-REML random-intercept ANCOVA and three comparator analyses |
| `extra_cells.py`, `power_table.py` | Extra S3 cells and the analytic power approximation |
| `causal.py` | Study S4: IV (2SLS) dose estimator and interventional mediation with cluster bootstrap |
| `diagram.py`, `diagrams.py` | Grayscale block-diagram engine (refuses overlapping boxes or labels) and Figs. 1-2 |
| `plots.py` | Result figures 3-8 |
| `run_all.sh` | Runs everything in order |

```bash
pip install -r requirements.txt
bash run_all.sh
```

Outputs (`*.json`, `fig*.png`) are written to this folder. Seeds are fixed in each script, so reruns reproduce the
manuscript values. Tested with Python 3.13, NumPy 2.5, SciPy 1.18, scikit-learn 1.9 and Matplotlib 3.11.
