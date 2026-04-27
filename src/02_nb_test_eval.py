import pandas as pd
import numpy as np
import os
import sys
import time

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, average_precision_score
from sklearn.naive_bayes import GaussianNB

import mlflow
import mlflow.sklearn

import warnings
import argparse
import logging
from typing import *
from utils import *

import yaml
from namespaces import DA

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


def build_naive_bayes_model(priors: List[float]):
    model = GaussianNB(priors=priors)
    return model


def eval_metrics(actual, pred_labels, pred_probas):
    auc = roc_auc_score(actual, pred_probas)
    ap = average_precision_score(actual, pred_probas)
    recall = recall_score(actual, pred_labels)
    precision = precision_score(actual, pred_labels)
    f1 = recall_score(actual, pred_labels)
    matthews_corr = matthews_corrcoef(actual, pred_labels)
    rc_at_127 = recall_at_top_n_percent(actual, pred_probas, 1.27)
    pr_at_127 = precision_at_top_n_percent(actual, pred_probas, 1.27)
    return auc, ap, recall, precision, f1, matthews_corr, rc_at_127, pr_at_127


if __name__ == "__main__":
    # 1. Setup Arguments
    parser = argparse.ArgumentParser()
    # Accept a string representing multiple sets of priors
    parser.add_argument("--feat_augment", action='store_true', default=False)
    args = parser.parse_args()

    FEAT_AUGMENT = args.feat_augment

    # ─────────────────────────────────────────────────────────────────────────────
    # Logger
    # ─────────────────────────────────────────────────────────────────────────────

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh  = logging.FileHandler(os.path.join(DA.paths.log, f'{time.time()}.log'))
    fh.setLevel(logging.DEBUG)
    ch  = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    logger.info(args)

    # load best params
    logger.info(f"Feature Augmentation: {FEAT_AUGMENT}")
    if FEAT_AUGMENT:
        # load best params for feat augment
        with open('hyperparameter_tuning/nb_best_params.yaml', 'r') as f:
            best_params = yaml.safe_load(f)
    else:
        # load best params for no feat augment
        with open('hyperparameter_tuning/nb_nofeat_best_params.yaml', 'r') as f:
            best_params = yaml.safe_load(f)

    
    best_params['priors'] = eval(best_params['priors'])  # Convert string back to list
    
    # 2. Load data ONCE
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(data_path=DA.paths.output_data_ml, feat_augment=args.feat_augment)
    
    # 3. Set MLflow Experiment
    EXPERIMENT_NAME = f"nb-test"
    if not FEAT_AUGMENT:
        EXPERIMENT_NAME += "-nofeat"

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        mlflow.log_params(best_params)

        # create model
        logger.info(f"Best Hyperparameters: {best_params}")
        model = build_naive_bayes_model(priors=best_params['priors'])
        # fit model
        model.fit(X_train, y_train.values.ravel())
        # predict on val set
        pred_probas = model.predict_proba(X_val)[:, 1]
        pred_labels = model.predict(X_val)
        # evaluate metrics
        auc, ap, recall, precision, f1, mcc, rc_at_127, pr_at_127 = eval_metrics(y_val, pred_labels, pred_probas)
        logger.info(f"Validation Metrics: AUC={auc:.4f}, AP={ap:.4f}, Recall={recall:.4f}, Precision={precision:.4f}, F1={f1:.4f}, Matthews Corr={mcc:.4f}, Recall@1.27%={rc_at_127:.4f}, Precision@1.27%={pr_at_127:.4f}")
        mlflow.log_metrics({
            "val_auc": auc,
            "val_ap": ap,
            "val_recall": recall,
            "val_precision": precision,
            "val_f1": f1,
            "val_matthews_corr": mcc,
            "val_recall_at_127": rc_at_127,
            "val_precision_at_127": pr_at_127
        })


        # predict on test set
        pred_probas = model.predict_proba(X_test)[:, 1]
        pred_labels = model.predict(X_test)
        # evaluate metrics
        auc, ap, recall, precision, f1, mcc, rc_at_127, pr_at_127 = eval_metrics(y_test, pred_labels, pred_probas)
        logger.info(f"Test Metrics: AUC={auc:.4f}, AP={ap:.4f}, Recall={recall:.4f}, Precision={precision:.4f}, F1={f1:.4f}, Matthews Corr={mcc:.4f}, Recall@1.27%={rc_at_127:.4f}, Precision@1.27%={pr_at_127:.4f}")
        mlflow.log_metrics({
            "test_auc": auc,
            "test_ap": ap,
            "test_recall": recall,
            "test_precision": precision,
            "test_f1": f1,
            "test_matthews_corr": mcc,
            "test_recall_at_127": rc_at_127,
            "test_precision_at_127": pr_at_127,
        })