#!/usr/bin/env bash
# Reproduces every number and figure in the manuscript (synthetic data only).
set -e
python3 run_mqs.py            # S1 simulation -> s1_raw.json
python3 analyse_s1.py         # S1 analysis -> s1.json
python3 negyaw.py             # S1 negative-yaw sweep -> s1_negyaw.json
python3 controller.py 7.406   # S2 controller; sigma_s = residual SD of the score in s1.json -> s2.json
python3 trialsim.py 4000      # S3 trial operating characteristics -> s3.json
python3 extra_cells.py        # S3 extra cells for delta = 1.5
python3 power_table.py        # analytic vs simulated power -> s3_table.json
python3 causal.py 1000        # S4 dose and mediation estimators -> s4.json
python3 diagrams.py           # Figs. 1-2
python3 plots.py              # Figs. 3-8
