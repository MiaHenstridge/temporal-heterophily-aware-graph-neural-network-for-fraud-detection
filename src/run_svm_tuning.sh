#!/bin/bash

feat_augment=(true false) # Set to true or false

for f in "${feat_augment[@]}"; do
    # Logic to determine the flag string
    if [ $f = true ]; then
        augment_flag="--feat_augment"
    else
        augment_flag=""
    fi

    python src/03_svm_calibrated_tuning.py \
        --c_values 0.1 1.0 10.0 \
        --loss_values squared_hinge \
        --weights "0.01,0.99|0.02,0.98|0.05,0.95|0.1,0.9" \
        $augment_flag
done
