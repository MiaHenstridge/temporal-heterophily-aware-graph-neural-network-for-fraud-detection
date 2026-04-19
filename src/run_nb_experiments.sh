#!/bin/bash

feat_augment=(true false) # Set to true or false

for f in "${feat_augment[@]}"; do
    # Logic to determine the flag string
    if [ $f = true ]; then
        augment_flag="--feat_augment"
    else
        augment_flag=""
    fi

    python src/02_nb_tuning.py \
        --priors_sweep "0.99,0.01|0.98,0.02|0.95,0.05|0.90,0.10|0.50,0.50" \
        $augment_flag
done
