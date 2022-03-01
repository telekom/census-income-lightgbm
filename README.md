# Census-Income with LightGBM and Optuna

This project uses the [census income data](https://archive-beta.ics.uci.edu/ml/datasets/census+income) and
fits [LightGBM](https://lightgbm.readthedocs.io/) models on it.

It is not intended to bring super good results, but rather as a demo to show the interaction between 
[LightGBM](https://lightgbm.readthedocs.io/), [Optuna](https://optuna.readthedocs.io/en/stable/index.html) and 
[HPOflow](https://github.com/telekom/HPOflow). The usage of HPOflow is optional and can be removed if wanted.

This work can be understood as a template for other projects.

- `simple_train.py`: do hyperparameter search with Optuna
- `save_train.py`: fit LightGBM on full dataset with best hyperparameter-set

## Usage

1. create and activate a new Python environment (for example with conda)
2. install the dependencies: `pip install -r requirements.txt`
3. execute `preprocess.ipynb` to load and preprocess the data
4. start the hyperparameter optimization with `python simple_train.py`
5. wait a few minutes
6. execute `optuna_vis.ipynb` to view the results (can be made in parallel while the optimization is still running)
7. also look at the graphics in the plots directory

## Still To-do

- provide a script for final model building
- evaluate final model on the test data
- use [SHAP (SHapley Additive exPlanations)](https://github.com/slundberg/shap) to calculate feature importance

## Licensing

Copyright (c) 2022 Philip May, Deutsche Telekom AG

Licensed under the **MIT License** (the "License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License by reviewing the file
[LICENSE](https://github.com/telekom/census-income-lightgbm/blob/main/LICENSE) in the repository.
