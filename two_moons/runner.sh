#!/bin/bash

python experiment.py --multirun training.loss=smcwake_naive,smcwake_a,smcwake_b,smcwake_c training.epochs=50000 data.prior_broad=false training.device=cuda:5
python experiment.py --multirun training.loss=wake,defensive_wake training.epochs=50000 data.prior_broad=false training.device=cuda:6
python plots.py --multirun plots.losses=['wake','defensive_wake','smcwake_a','smcwake_b','smcwake_c'],['wake','defensive_wake','smcwake_a']
