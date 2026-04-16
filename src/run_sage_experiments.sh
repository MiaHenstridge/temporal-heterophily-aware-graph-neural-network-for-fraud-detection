#!/bin/bash

# Define parameter arrays
model=sage
lr=1e-4
bs=512
epochs=200
node_dim=128
neighbors=(5 10)
layers=(2 3 4)

total_runs=$((${#layers[@]} * ${#neighbors[@]}))

# Initialize run counter
run_id=0

for l in "${layers[@]}"; do
    for n in "${neighbors[@]}"; do
        # Create the run prefix
        prefix="run$run_id"
        
        echo "Running: $prefix ($((run_id + 1))/$total_runs) | layer=$l, neighbor=$n"
        
        # Pass the prefix as an argument to your script
        python src/05_static_graph_tuning.py \
            --model "$model"\
            --prefix "$prefix" \
            --lr "$lr" \
            --bs "$bs" \
            --n_epoch "$epochs" \
            --feat_augment \
            --loss focal \
            --node_dim "$node_dim"\
            --n_layer "$l" \
            --n_neighbor "$n" \
            --early_stop_higher_better
        
        # Increment counter for the next iteration
        ((run_id++))
    done
done