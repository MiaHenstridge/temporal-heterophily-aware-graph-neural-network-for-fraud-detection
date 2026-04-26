#!/bin/bash

lr=1e-4
bs=512
epochs=200
node_dim=128
time_dim=128
smp_layers=1
neighbors=10
duration=365
hops=1
feat_augment=false
max_rounds=10
seeds=(123 456 789 101112 131415 161718 192021 222324 252627 282930)

# Fixed variable name from seed to seeds
total_runs=$((${#seeds[@]}))

run_id=0

for s in "${seeds[@]}"; do
    
    if [ "$feat_augment" = true ]; then
        augment_flag="--feat_augment"
    else
        augment_flag=""
    fi

    prefix="run$run_id"
    
    echo "Running: $prefix ($((run_id + 1))/$total_runs) | aug=$feat_augment, hops=$hops, duration=$duration, neighbor=$neighbors, seed=$s"
    
    # Cleaned up the backslashes and removed trailing spaces
    python src/07_thegcn_tuning_with_sampler.py \
        --prefix "$prefix" \
        --lr "$lr" \
        --bs "$bs" \
        --n_epoch "$epochs" \
        --loss focal \
        --node_dim "$node_dim" \
        --time_dim "$time_dim" \
        --n_layer "$smp_layers" \
        --duration "$duration" \
        --n_hop "$hops" \
        --n_neighbor "$neighbors" \
        --early_stop_higher_better \
        --seed "$s" \
        --max_round "$max_rounds"\
        $augment_flag
    
    ((run_id++))
done