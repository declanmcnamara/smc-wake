#!/bin/bash

python experiment.py training.loss=smcwake_og training.refresh_time=10000000 smc.K=10000 training.epochs=100000 training.full_rank=true data.n_pts=10 smc.num_init=1 data.k=50 data.n=100 data.tau=1 training.device=cuda:3
python experiment.py training.loss=smcwake_og training.refresh_time=10000000 smc.K=100 training.epochs=100000 training.full_rank=true data.n_pts=10 smc.num_init=100 data.k=50 data.n=100 data.tau=1 training.device=cuda:3
python plotter.py