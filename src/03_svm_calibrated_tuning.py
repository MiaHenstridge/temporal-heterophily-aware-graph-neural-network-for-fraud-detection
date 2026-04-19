import pandas as pd
import numpy as np
import os

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, average_precision_score
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid

import mlflow
import mlflow.sklearn

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


def build_svm_model(C=1.0, loss='hinge', class_weight={0: 0.5, 1: 0.5}, random_state=42):
    pipeline = Pipeline(
        [
            ('scaler', MinMaxScaler()),
            ('svm', CalibratedClassifierCV(
                LinearSVC(penalty='l2', 
                              C=C,
                              loss=loss,
                              class_weight=class_weight,
                              random_state=random_state)
            ))
        ]
    )
    return pipeline


def parse_weight_arg(input_str):
    """
    Parses a string like '0.1,0.9|0.2,0.8' 
    into a list of dicts: [{0:0.1, 1:0.9}, {0:0.2, 1:0.8}]
    """
    sets = input_str.split('|')
    return [{i: v for i, v in enumerate(map(float, s.split(',')))} for s in sets]


if __name__ == "__main__":
    RANDOM_SEED = 1111

    # define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--c_values", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    parser.add_argument("--loss_values", type=str, nargs="+", default=["squared_hinge"])
    parser.add_argument("--weights", type=str, default="0.01,0.99|0.02,0.98|0.05,0.95|0.1,0.9")
    parser.add_argument("--feat_augment", action='store_true', default=False)
    args = parser.parse_args()

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(data_path=DA.paths.output_data_ml, feat_augment=args.feat_augment)
    
    # 1. Define the Parameter Grid including Tolerance
    param_grid = {
        'C': args.c_values,
        'loss': args.loss_values,
        'class_weight': parse_weight_arg(args.weights)
    }
    
    # Create all combinations (Cartesian product)
    grid = list(ParameterGrid(param_grid))
    
    mlflow.set_experiment("svm_calibrated_grid_search")

    print(f"Total runs to execute: {len(grid)}")

    # 2. Loop through the grid
    for params in grid:
        # Improved run naming for the UI
        w_val = list(params['class_weight'].values())[1] # Use the minority class weight for the label
        run_name = f"C={params['C']}_W1={w_val}_loss={params['loss']}"

        with mlflow.start_run(run_name=run_name):           
            # build model
            model = build_svm_model(
                C=params['C'], 
                loss=params['loss'],
                class_weight=params['class_weight'],
                random_state=RANDOM_SEED,
            )
            
            model.fit(X_train, y_train.values.ravel())
            
            # Eval
            scores = model.predict_proba(X_val)[:,1]
            labels = model.predict(X_val)
            auc, ap, recall, precision, f1, mcc = eval_metrics(y_val, labels, scores)
            
            # 3. Log everything
            mlflow.log_params({
                "C": params['C'],
                "loss": params['loss'],
                "class_weight": str(params['class_weight']),
                "feat_augment": args.feat_augment
            })
            
            mlflow.log_metrics({
                "auc": auc,
                "ap": ap,
                "recall": recall,
                "precision": precision,
                "f1": f1,
                "mcc": mcc
            })
            
            mlflow.sklearn.log_model(model, "model")
            print(f"Finished: {run_name} \nAUC: {auc:.4f} | AP: {ap:.4f} | Recall: {recall:.4f} | Precision: {precision:.4f} | F1: {f1:.4f} | MCC: {mcc:.4f}")

    print("\nRuns complete. Run 'mlflow ui' to compare the runs.")