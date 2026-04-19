#!/bin/bash

# Define parameter arrays
lr=1e-4
bs=512
epochs=200
node_dim=128
time_dim=128
neighbors=(5 10)
durations=(0 730 365)
layers=(2 3 4)
feat_aug_options=(true false)

# Total runs calculation
total_runs=$((${#feat_aug_options[@]} * ${#layers[@]} * ${#durations[@]} * ${#neighbors[@]}))

# Initialize run counter at 0
run_id=0

for feat_augment in "${feat_aug_options[@]}"; do
    
    # Logic to determine the flag string
    if [ "$feat_augment" = true ]; then
        augment_flag="--feat_augment"
    else
        augment_flag=""
    fi

    for l in "${layers[@]}"; do
        for d in "${durations[@]}"; do
            for n in "${neighbors[@]}"; do
                # Create the run prefix
                prefix="run$run_id"
                
                echo "Running: $prefix ($((run_id + 1))/$total_runs) | aug=$feat_augment, layer=$l, duration=$d, neighbor=$n"
                
                # Execute python script
                python src/07_thegcn_tuning_with_sampler.py \
                    --prefix "$prefix" \
                    --lr "$lr" \
                    --bs "$bs" \
                    --n_epoch "$epochs" \
                    --loss focal \
                    --node_dim "$node_dim" \
                    --time_dim "$time_dim" \
                    --n_layer "$l" \
                    --duration "$d" \
                    --n_neighbor "$n" \
                    --early_stop_higher_better \
                    $augment_flag
                
                # Increment counter
                ((run_id++))
            done
        done
    done
done