#!/bin/bash

# Define parameter arrays
lr=1e-4
bs=512
epochs=200
heads=4
node_dim=128
time_dim=128
neighbors=(5 10)
durations=(0 730 365)
layers=(2 3 4)
# Added array for augmentation options
feat_aug_options=(true false) 

# Total runs now includes the augmentation dimension
total_runs=$((${#feat_aug_options[@]} * ${#layers[@]} * ${#durations[@]} * ${#neighbors[@]}))

# Initialize run counter
run_id=0

# Outer loop for augmentation
for feat_augment in "${feat_aug_options[@]}"; do
    
    # Determine the flag string dynamically for each iteration
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
                
                # Updated echo to show augmentation status
                echo "Running: $prefix ($((run_id + 1))/$total_runs) | aug=$feat_augment, layer=$l, duration=$d, neighbor=$n"
                
                # Execute python script
                python src/06_tgat_tuning_with_sampler.py \
                    --prefix "$prefix" \
                    --lr "$lr" \
                    --bs "$bs" \
                    --n_epoch "$epochs" \
                    --loss focal \
                    --node_dim "$node_dim" \
                    --time_dim "$time_dim" \
                    --n_head "$heads" \
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