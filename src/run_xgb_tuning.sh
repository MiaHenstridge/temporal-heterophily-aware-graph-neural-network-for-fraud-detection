#!/bin/bash

feat_augment=(true false) # Set to true or false

for f in "${feat_augment[@]}"; do
    # Logic to determine the flag string
    if [ $f = true ]; then
        augment_flag="--feat_augment"
    else
        augment_flag=""
    fi

    python src/04_xgboost_tuning.py \
        --direction maximize \
        --n_trials 10 \
        $augment_flag
done
