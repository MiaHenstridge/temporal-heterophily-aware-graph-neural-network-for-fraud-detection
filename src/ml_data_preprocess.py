import pandas as pd
import numpy as np
import os
from namespaces import DA


# DATA_PATH = '../datasets/DGraphFin/'
os.makedirs(DA.paths.data, exist_ok=True)


# OUTPUT_DATA_PATH = '../processed_data'
os.makedirs(DA.paths.output_data, exist_ok=True)

# OUTPUT_DATA_ML = os.path.join(OUTPUT_DATA_PATH, 'baseline_ml')
os.makedirs(DA.paths.output_data_ml, exist_ok=True)


# 1. Load the dataset
data = np.load(os.path.join(DA.paths.data, 'dgraphfin.npz'))

# 2. Extract node features and labels
x = data['x']
y = data['y']

# 3. relabel y for binary classification
print(f"Original classes: {np.unique(y)}")

# relabel 1 to 1, everything else to 0
y = np.where(y==1, 1, 0)
print(f"Classes after relabeling: {np.unique(y)}")

# 4. Split data using the provided masks
train_mask = data['train_mask']
valid_mask = data['valid_mask']
test_mask = data['test_mask']

x_train, y_train = x[train_mask], y[train_mask]
x_val, y_val = x[valid_mask], y[valid_mask]
x_test, y_test = x[test_mask], y[test_mask]

print(f"Total nodes: {x.shape[0]}")
print(f"Fraud nodes: {np.sum(y==1)}")
print(f"Non-fraud cases: {np.sum(y==0)}")

# 5. save to a single compressed file
np.savez_compressed(os.path.join(DA.paths.output_data_ml, 'dgraphfin_processed.npz'),
                   x_train=x_train, y_train=y_train,
                   x_val=x_val, y_val=y_val,
                   x_test=x_test, y_test=y_test)