#!/bin/bash

# Define parameter arrays
lr=1e-4
bs=512
epochs=200
node_dim=128
time_dim=128
smp_layers=1
neighbors=(3 5 10)
durations=(730 365 180 90 30)
hops=(1 2 3)
feat_aug_options=(true false)

# Total runs calculation
total_runs=$((${#feat_aug_options[@]} * ${#hops[@]} * ${#durations[@]} * ${#neighbors[@]}))

# Initialize run counter at 0
run_id=0

for feat_augment in "${feat_aug_options[@]}"; do
    
    # Logic to determine the flag string
    if [ "$feat_augment" = true ]; then
        augment_flag="--feat_augment"
    else
        augment_flag=""
    fi

    for h in "${hops[@]}"; do
        for d in "${durations[@]}"; do
            for n in "${neighbors[@]}"; do
                # Create the run prefix
                prefix="run$run_id"
                
                echo "Running: $prefix ($((run_id + 1))/$total_runs) | aug=$feat_augment, hops=$h, duration=$d, neighbor=$n"
                
                # Execute python script
                python src/07_thegcn_tuning_with_sampler.py \
                    --prefix "$prefix" \
                    --lr "$lr" \
                    --bs "$bs" \
                    --n_epoch "$epochs" \
                    --loss focal \
                    --node_dim "$node_dim" \
                    --time_dim "$time_dim" \
                    --n_layer "$smp_layers" \
                    --duration "$d" \
                    --n_hop "$h"\
                    --n_neighbor "$n" \
                    --early_stop_higher_better \
                    $augment_flag
                
                # Increment counter
                ((run_id++))
            done
        done
    done
done