# SMC-Wake

This repository contains code to replicate the experiments from the paper *Sequential Monte Carlo for Inclusive KL Minimization in Amortized Variational Inference* at the International Conference on Artificial Intelligence and Statistics (AISTATS), 2024.

To get started, create a fresh virtual environment and install required packages as follows:
```
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For each experiment, adjust the `dir` field in the relevant config file in `conf` to be the directory of this repo. Thereafter, results for any experiment of choice can be replicated by e.g.

```
cd two_moons
chmod +x runner.sh
./runner.sh
```




