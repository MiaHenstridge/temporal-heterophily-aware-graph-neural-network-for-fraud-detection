import pandas as pd
import numpy as np
import os

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.naive_bayes import GaussianNB

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


# DATA_PATH = './datasets/DGraphFin/'
# os.makedirs(DATA_PATH, exist_ok=True)

# OUTPUT_DATA_PATH = './processed_data'
# os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

# OUTPUT_DATA_ML = os.path.join(OUTPUT_DATA_PATH, 'baseline_ml')
# os.makedirs(OUTPUT_DATA_ML, exist_ok=True)

os.chdir('/home/mai/notebooks/final_thesis/')
print(f"Current working dir: {os.getcwd()}")


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


def build_naive_bayes_model(priors: List[float]):
    model = GaussianNB(priors=priors)
    return model


def eval_metrics(actual, pred_labels, pred_probas):
    auc = roc_auc_score(actual, pred_probas)
    recall = recall_score(actual, pred_labels)
    precision = precision_score(actual, pred_labels)
    f1 = recall_score(actual, pred_labels)
    matthews_corr = matthews_corrcoef(actual, pred_labels)
    return auc, recall, precision, f1, matthews_corr


def parse_priors_input(input_str):
    """
    Parses a string like '0.9,0.1|0.8,0.2' 
    into a list of lists: [[0.9, 0.1], [0.8, 0.2]]
    """
    sets = input_str.split('|')
    return [list(map(float, s.split(','))) for s in sets]

if __name__ == "__main__":
    # 1. Setup Arguments
    parser = argparse.ArgumentParser()
    # Accept a string representing multiple sets of priors
    parser.add_argument("--priors_sweep", type=str, 
                        default="0.99,0.01|0.98,0.02|0.95,0.05|0.90,0.10|0.50,0.50",
                        help="Sets of priors separated by | (e.g. '0.9,0.1|0.8,0.2')")
    args = parser.parse_args()
    
    # Parse the string into a list of lists
    priors_to_test = parse_priors_input(args.priors_sweep)
    
    # 2. Load data ONCE
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(data_path=DA.paths.output_data_ml)
    
    # 3. Set MLflow Experiment
    mlflow.set_experiment(experiment_name="nb_priors_tuning")

    print(f"Starting runs for {len(priors_to_test)} prior configurations...")

    # 4. The Loop
    for p_set in priors_to_test:
        # Create a descriptive run name based on the priors
        run_name = f"Priors_{p_set[0]}_{p_set[1]}"
        
        with mlflow.start_run(run_name=run_name):
            # Build and Fit
            model = build_naive_bayes_model(priors=p_set)
            # GaussianNB expects y to be 1D
            model.fit(X_train, y_train.values.ravel()) 
            
            # Predict
            pred_probs = model.predict_proba(X_val)[:, 1]
            pred_labels = model.predict(X_val)
            
            # Metrics
            auc, recall, precision, f1, matthews_corr = eval_metrics(
                y_val, pred_labels=pred_labels, pred_probas=pred_probs
            )
            
            # Log to MLflow
            # We log the whole list as a string so MLflow displays it clearly
            mlflow.log_param("priors", str(p_set))
            mlflow.log_metric("auc", auc)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("mcc", matthews_corr)
            
            # Log the model
            mlflow.sklearn.log_model(model, "model")
            
            print(f"Finished: {p_set} | AUC: {auc:.4f} | Recall: {recall:.4f} | Precision: {precision:.4f} | F1: {f1:.4f} | MCC: {matthews_corr:.4f}")

    print("\nRuns complete. Run 'mlflow ui' to compare the runs.")