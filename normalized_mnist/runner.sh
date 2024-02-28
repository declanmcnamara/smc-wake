#!/bin/bash

python data.py
python experiment.py --multirun training.device='cuda:1' training.loss=wake,smcwake2 training.epochs=50000 data.latent_dim=16 training.lr=5e-4 smc.K=100 training.prior_refresh=true training.mb_size=32 data.prior_dispersion=1
python results.py training.loss=wake data.latent_dim=16 data.prior_dispersion=1 training.device='cpu'







