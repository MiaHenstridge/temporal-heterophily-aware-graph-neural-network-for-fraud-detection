import pandas as pd
import numpy as np
import os

from sklearn.metrics import (
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    matthews_corrcoef
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
# from sklearn.model_selection import ParameterGrid
from xgboost import XGBClassifier
import optuna

import mlflow
import mlflow.xgboost

import warnings
import argparse
import logging
from typing import *

from namespaces import DA


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


os.chdir('/home/mai/notebooks/final_thesis/')
print(f"Current working directory: {os.getcwd()}")


def load_data(data_path):
    data = np.load(os.path.join(data_path, 'dgraphfin_processed.npz'))
    # train
    X_train = pd.DataFrame(data['x_train'])
    y_train = pd.DataFrame(data['y_train'])
    # val
    X_val = pd.DataFrame(data['x_val'])
    y_val = pd.DataFrame(data['y_val'])
    # test
    X_test = pd.DataFrame(data['x_test'])
    y_test = pd.DataFrame(data['y_test'])
    return X_train, X_val, X_test, y_train, y_val, y_test


def eval_metrics(actual, pred_labels, pred_probas):
    auc = roc_auc_score(actual, pred_probas)
    recall = recall_score(actual, pred_labels)
    precision = precision_score(actual, pred_labels)
    f1 = recall_score(actual, pred_labels)
    matthews_corr = matthews_corrcoef(actual, pred_labels)
    return auc, recall, precision, f1, matthews_corr


def objective(trial):
    param_grid_ = {
        'verbose': 0,
        'random_seed': RANDOM_SEED,
        'n_jobs': -1,
        'max_depth': trial.suggest_int('depth', 4, 6),
        'max_leaves': trial.suggest_int('max_leaves', 64, 256, 16),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 3000, 10000, 100),
        'subsample': trial.suggest_loguniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_loguniform('colsample_bytree', 0.5, 1.0),
        'colsample_bylevel': trial.suggest_loguniform('colsample_bylevel', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'missing': -1,
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 30.0, 300.0, log=True),
        'max_delta_step': trial.suggest_int("max_delta_step", 1, 10),
        'min_child_weight': trial.suggest_int("min_child_weight", 1, 25),
        'gamma': trial.suggest_float("gamma", 0, 10),
    }

    with mlflow.start_run(nested=True):
        # create a model with the params
        model = XGBClassifier(**param_grid_)
    
        model.fit(X_train, y_train)
    
        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)
    
        auc, recall, precision, f1, mcc = eval_metrics(y_val, y_pred, y_prob)
        # log params
        mlflow.log_params(param_grid_)
        # log metrics
        mlflow.log_metric("auc", float(auc))
        mlflow.log_metric("recall", float(recall))
        mlflow.log_metric("precision", float(precision))
        mlflow.log_metric("f1", float(f1))
        mlflow.log_metric("mcc", float(mcc))
        # log model
        mlflow.xgboost.log_model(model, "model")
        
    return auc


if __name__ == '__main__':
    # set random seed
    RANDOM_SEED = 1111

    # define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", type=str, default="maximize", required=False)
    parser.add_argument("--n_trials", type=int, default=20, required=False)
    args = parser.parse_args()

    # Load data
    X_train, X_val, _, y_train, y_val, _ = load_data(data_path=DA.paths.output_data_ml)

    # create experiment
    mlflow.set_experiment("xgb_optuna_tuning")
    with mlflow.start_run(run_name="xgb_optuna_study"):
        study = optuna.create_study(
            study_name='xgb_study',
            direction=args.direction,
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.HyperbandPruner()
        )

        try:
            print(f"Total runs to execute: {args.n_trials}")
            study.optimize(objective, n_trials=args.n_trials)
        except:
            raise

        # log best results at parent level
        mlflow.log_params(study.best_params)
        mlflow.log_metric("auc", study.best_value)
        
    print("\nRuns complete. Run 'mlflow ui' to compare the runs.")