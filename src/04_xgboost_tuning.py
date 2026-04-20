import pandas as pd
import numpy as np
import os

from sklearn.metrics import (
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    matthews_corrcoef,
    average_precision_score
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


def load_data(data_path, feat_augment=True):
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

    if not feat_augment:
        X_train = X_train.iloc[:, :17]
        X_val = X_val.iloc[:, :17]
        X_test = X_test.iloc[:, :17]
    return X_train, X_val, X_test, y_train, y_val, y_test


def eval_metrics(actual, pred_labels, pred_probas):
    auc = roc_auc_score(actual, pred_probas)
    ap = average_precision_score(actual, pred_probas)
    recall = recall_score(actual, pred_labels)
    precision = precision_score(actual, pred_labels)
    f1 = recall_score(actual, pred_labels)
    matthews_corr = matthews_corrcoef(actual, pred_labels)
    return auc, ap, recall, precision, f1, matthews_corr


def objective(trial):
    param_grid_ = {
        'verbose': 0,
        'random_seed': RANDOM_SEED,
        'n_jobs': -1,
        'max_depth': trial.suggest_int('depth', 3, 5),
        'max_leaves': trial.suggest_int('max_leaves', 64, 256, 16),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 3000, 7000, 100),
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

    with mlflow.start_run():  # ← no longer nested
        model = XGBClassifier(**param_grid_)
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)

        auc, ap, recall, precision, f1, mcc = eval_metrics(y_val, y_pred, y_prob)

        mlflow.log_params(param_grid_)
        mlflow.log_param('feat_augment', FEAT_AUGMENT)  # ← log here instead
        mlflow.log_metric("auc", float(auc))
        mlflow.log_metric("ap", float(ap))
        mlflow.log_metric("recall", float(recall))
        mlflow.log_metric("precision", float(precision))
        mlflow.log_metric("f1", float(f1))
        mlflow.log_metric("mcc", float(mcc))
        mlflow.xgboost.log_model(model, artifact_path="model")

    return ap


if __name__ == '__main__':
    RANDOM_SEED = 1111

    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", type=str, default="maximize", required=False)
    parser.add_argument("--n_trials", type=int, default=20, required=False)
    parser.add_argument("--feat_augment", action='store_true', default=False)
    args = parser.parse_args()

    FEAT_AUGMENT = args.feat_augment  # ← make accessible to objective()

    X_train, X_val, _, y_train, y_val, _ = load_data(
        data_path=DA.paths.output_data_ml, feat_augment=args.feat_augment
    )

    mlflow.set_experiment("xgb_optuna_tuning")

    # ← no more parent start_run wrapper
    study = optuna.create_study(
        study_name='xgb_study',
        direction=args.direction,
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.HyperbandPruner()
    )

    print(f"Total runs to execute: {args.n_trials}")
    study.optimize(objective, n_trials=args.n_trials)

    print("\nRuns complete. Run 'mlflow ui' to compare the runs.")