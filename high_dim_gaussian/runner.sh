#!/bin/bash

python experiment.py training.loss=msc training.refresh_time=1 smc.K=100 training.epochs=500000 training.full_rank=true data.n_pts=50 smc.num_init=1 data.k=50 data.n=100 data.tau=1 training.device=cuda:2 seed=578493
python experiment.py training.loss=smcwake_pimh training.refresh_time=1 smc.K=100 training.epochs=40000 training.full_rank=true data.n_pts=50 smc.num_init=1 data.k=50 data.n=100 data.tau=1 training.device=cuda:2 seed=578493
python results.py smc.K=100 smc.run=false training.epochs=500000 training.full_rank=true data.n_pts=50 smc.num_init=1 data.k=50 data.n=100 data.tau=1 training.device=cpu seed=578493






