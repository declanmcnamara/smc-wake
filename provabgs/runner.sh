#!/bin/bash

python experiment.py --multirun training.loss=smcwake_a training.device='cuda:1' smc.K=10,100 smc.refresh_rate=50
python plot_mcmc.py training.loss=smcwake_a smc.K=10 smc.refresh_rate=50 smc.run=false
python plot_mcmc.py training.loss=smcwake_a smc.K=100 smc.refresh_rate=50 smc.run=false
