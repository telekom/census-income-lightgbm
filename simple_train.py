from functools import partial
import math

import numpy as np
import pandas as pd
import optuna
import lightgbm
from hpoflow import SignificanceRepeatedTrainingPruner
from sklearn.model_selection import StratifiedKFold


def fit(trial, train_x, train_y, val_x, val_y):
    # Optional: add info about categorical_feature (in Dataset or train params)
    # Optional: should "fnlwgt" be a categorical_feature?
    # example:  categorical_feature=["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
    train_data = lightgbm.Dataset(train_x, label=train_y)
    test_data = lightgbm.Dataset(val_x, label=val_y)

    # will be filled by LightGBM
    evals_result = {}    
    
    params = {
        "objective":"binary",
        "metric": "auc",  # we might want to use F1 score here - mltb can be used to do that - see https://github.com/PhilipMay/mltb#module-lightgbm
        "verbose":-1,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5),  # default 0.1 constraint > 0.0        
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 30),  # default 20 constraint >= 0
        "max_bin": trial.suggest_int("max_bin", 100, 2000),  # default 255 constraint > 1
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.01, 1.0),  # default 1.0 constraint 0.0 < bagging_fraction <= 1.0
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),  # default 0.0 constraint >= 0.0

        # this is optional
        # "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 80.0),
        # "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 80.0),
        # "extra_trees": trial.suggest_categorical("extra_trees", [True, False])
    }    
    
    # there is a special dependency between max_depth and num_leaves
    # see https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html#tune-parameters-for-the-leaf-wise-best-first-tree
    max_depth = trial.suggest_int("max_depth", 1, 9)  # default -1 
    params["max_depth"] = max_depth
    num_leaves = trial.suggest_int("num_leaves", 2, 2**max_depth)  # default 31 constraint 1 < num_leaves <= 131072
    params["num_leaves"] = num_leaves
    
    # this is optional
    # if trial.suggest_categorical("do_bagging", [True, False]):
    #     params["bagging_freq"] = trial.suggest_int("bagging_freq", 1, 10)
    #     params["feature_fraction"] = trial.suggest_float("feature_fraction", 0.1, 1.0)
    
    # fir the model
    bst = lightgbm.train(
        params, 
        train_data, 
        num_boost_round=trial.suggest_int("num_boost_round", 10, 200),  # default 100 constraint >= 0 (although 0 of course makes no sense)
        valid_sets=[test_data], 
        callbacks=[lightgbm.record_evaluation(evals_result)],
    )    

    # get the result result of the last boosing round
    result = evals_result["valid_0"]["auc"][-1]
    
    # make sure nothing crashes
    if result < 0 or math.isnan(result):
        result = 0.0
    
    return result
    
def objective(trial, data):
    # list of results of CV
    results = []
    
    # 10 fold CV 
    # to ensure that the results of the different HP evaluations are comparable, we use a seed here
    skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    for fold_id, (train_index, val_index) in enumerate(skf.split(data, data["label"])):  # make sure to have stratified splits
        
        # create train data from train_index
        train_x = data.iloc[train_index].copy()
        train_x.drop("label", axis=1, inplace=True)
        train_y = data.iloc[train_index]["label"].to_list()
        assert len(train_x) == len(train_y)
        
        # create train data from val_index
        val_x = data.iloc[val_index].copy()
        val_x.drop("label", axis=1, inplace=True)
        val_y = data.iloc[val_index]["label"].to_list()
        assert len(val_x) == len(val_y)

        # fit the model
        result = fit(trial, train_x, train_y, val_x, val_y)

        # collect results of the CV
        results.append(result)
        
        # set the results to optuna as user attribute for tracking
        trial.set_user_attr("results", str(results))
    
        # optional: this uses SignificanceRepeatedTrainingPruner to check if we can prune the cross validation
        # see https://telekom.github.io/HPOflow/doc/SignificanceRepeatedTrainingPruner.html#significancerepeatedtrainingpruner-doc
        trial.report(result, fold_id)
        if trial.should_prune():
            break    
    
    # return CV result
    return np.mean(results)
        
if __name__ == "__main__":
    # load data
    df = df = pd.read_csv("data/census_income_train.csv", low_memory=False)

    # create optuna study
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(multivariate=True),
        study_name="lightgbm_simple_01",
        
        # we use a locale DB here (a file)
        # if you want to do HP search on multiple machines use an external SQL db
        storage="sqlite:///optuna.db",
        # storage=f"postgresql://optuna:{OPTUNA_PASS}@{OPTUNA_HOSTNAME}/optuna",
        
        load_if_exists=True,
        direction="maximize",
        
        # optional: this can speed up HP search if fitting needs much time
        pruner=SignificanceRepeatedTrainingPruner(
            alpha=0.4,
            n_warmup_steps=3,
        ),
    )

    # optuna expects a function with only one parameter (trial)
    # this is a trick to pass the data anyway
    objective_partial = partial(objective, data=df)
    
    # we do not specify n_trials here
    # this means that the optimization is executed until it is canceled
    # that can be done by pressing ctrl + c
    # the optimization can just restarted
    # previous results are then simply loaded at the beginning from the DB
    study.optimize(objective_partial)
