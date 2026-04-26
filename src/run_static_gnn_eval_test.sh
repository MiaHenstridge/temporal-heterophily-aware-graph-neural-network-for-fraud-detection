#!/bin/bash

# Define parameter arrays
model=fagcn
lr=1e-4
bs=512
epochs=200
node_dim=128
layers=1
neighbors=5
feat_augment=false
max_rounds=5
seeds=(123 456 789 101112 131415 161718 192021 222324 252627 282930)

# Initialize run counter at 0 for each model
run_id=0

# Total runs per model (now includes the augmentation dimension)
total_runs=$((${#seeds[@]}))

for s in "${seeds[@]}"; do
    # Create a unique prefix for this model and run
    prefix="${model}_run$run_id"
       
    # Determine the flag string for each iteration
    if [ "$feat_augment" = true ]; then
        augment_flag="--feat_augment"
    else
        augment_flag=""
    fi
    
    echo "Running: $prefix ($((run_id + 1))/$total_runs) | model=$model, aug=$feat_augment | layer=$layers, neighbor=$neighbors, seed=$s"
    
    python src/05_static_graph_tuning.py \
        --model "$model" \
        --prefix "$prefix" \
        --lr "$lr" \
        --bs "$bs" \
        --n_epoch "$epochs" \
        --loss focal \
        --node_dim "$node_dim" \
        --n_layer "$layers" \
        --n_neighbor "$neighbors" \
        --early_stop_higher_better \
        --max_round "$max_rounds" \
        --seed "$s" \
        $augment_flag

    # Increment counter
    ((run_id++))
done