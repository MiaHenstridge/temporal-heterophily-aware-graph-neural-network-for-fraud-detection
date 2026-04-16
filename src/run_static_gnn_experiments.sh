#!/bin/bash

# Define parameter arrays
models=(sage gat gatv2 fagcn)
lr=1e-4
bs=512
epochs=200
node_dim=128
neighbors=(5 10)
layers=(2 3 4)

total_runs=$((${#layers[@]} * ${#neighbors[@]}))

for m in "${models[@]}"; do
    # Initialize run counter for this specific model
    run_id=0
    
    for l in "${layers[@]}"; do
        for n in "${neighbors[@]}"; do
            # Create a unique prefix for this model and this specific run
            prefix="${m}_run$run_id"
            
            echo "Running: $prefix ($((run_id + 1))/$total_runs) | model=$m | layer=$l, neighbor=$n"
            
            python src/05_static_graph_tuning.py \
                --model "$m" \
                --prefix "$prefix" \
                --lr "$lr" \
                --bs "$bs" \
                --n_epoch "$epochs" \
                --feat_augment \
                --loss focal \
                --node_dim "$node_dim" \
                --n_layer "$l" \
                --n_neighbor "$n" \
                --early_stop_higher_better
            
            # Increment counter for this model
            ((run_id++))
        done
    done
done