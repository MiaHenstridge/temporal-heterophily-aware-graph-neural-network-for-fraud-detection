#!/bin/bash

feat_augment=(true false) # Set to true or false
seeds=(123 456 789 101112 131415 161718 192021 222324 252627 282930)

for f in "${feat_augment[@]}"; do
    # Logic to determine the flag string
    if [ $f = true ]; then
        augment_flag="--feat_augment"
    else
        augment_flag=""
    fi
    for s in "${seeds[@]}"; do
        echo "Running XGBoost Test Eval | aug=$f, seed=$s"

        python src/04_xgboost_test_eval.py \
            --seed "$s" \
            $augment_flag
    done
done
