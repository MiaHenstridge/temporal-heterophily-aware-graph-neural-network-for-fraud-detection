#!/bin/bash

feat_augment=(true false) # Set to true or false

for f in "${feat_augment[@]}"; do
    # Logic to determine the flag string
    if [ $f = true ]; then
        augment_flag="--feat_augment"
    else
        augment_flag=""
    fi

    echo "Running XGBoost Test Eval | aug=$f, seed=$s"

    python src/02_nb_test_eval.py \
        $augment_flag
done
