# Census-Income with LightGBM and Optuna

This project uses the [census income data](https://archive-beta.ics.uci.edu/ml/datasets/census+income) and
fits [LightGBM](https://lightgbm.readthedocs.io/) models on it. We also calculare the feature importances
with [SHAP (SHapley Additive exPlanations)](https://github.com/slundberg/shap).

It is not intended to bring super good results, but rather as a demo to show the interaction between 
[LightGBM](https://lightgbm.readthedocs.io/), [Optuna](https://optuna.readthedocs.io/en/stable/index.html) and 
[HPOflow](https://github.com/telekom/HPOflow). The usage of HPOflow is optional and can be removed if wanted.

This work can be understood as a template for other projects.

## File Description

The scripts and notebooks should be executed in this order.

1. `preprocess.ipynb`: download, explore and preprocess the data
2. `simple_train.py`: do hyperparameter search with Optuna
3. `optuna_vis.ipynb`: print and visualize optuna results
4. `save_train.py`: fit LightGBM on full dataset with best hyperparameter-set - this is an extension of `simple_train.py` and adds the option to store the model with the best hyperparameter set - therefore there is a lot of redundancy with `simple_train.py`
5. `shap_values.ipynb`: calculate and visualize shap values / feature importance
6. `optuna.db`: this was intentionally placed in git to be able to visualize the results directly using `optuna_vis.ipynb`

## Usage

1. create and activate a new Python environment (for example with conda)
2. install the dependencies: `pip install -r requirements.txt`
3. execute `preprocess.ipynb` to load and preprocess the data
4. start the hyperparameter optimization with `python simple_train.py`
5. wait a few minutes
6. execute `optuna_vis.ipynb` to view the results (can be made in parallel while the optimization is still running)
7. also look at the graphics in the plots directory

## Licensing

Copyright (c) 2022 Philip May, Deutsche Telekom AG

Licensed under the **MIT License** (the "License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License by reviewing the file
[LICENSE](https://github.com/telekom/census-income-lightgbm/blob/main/LICENSE) in the repository.
