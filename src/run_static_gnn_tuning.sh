#!/bin/bash

# Define parameter arrays
models=(sage gat gatv2 fagcn)
lr=1e-4
bs=512
epochs=200
node_dim=128
neighbors=(3 5 10)
layers=(1 2 3)
feat_aug_options=(true false)

# Total runs per model (now includes the augmentation dimension)
total_runs=$((${#feat_aug_options[@]} * ${#layers[@]} * ${#neighbors[@]}))

for m in "${models[@]}"; do
    # Initialize run counter at 0 for each model
    run_id=0
    
    for feat_augment in "${feat_aug_options[@]}"; do
        
        # Determine the flag string for each iteration
        if [ "$feat_augment" = true ]; then
            augment_flag="--feat_augment"
        else
            augment_flag=""
        fi

        for l in "${layers[@]}"; do
            for n in "${neighbors[@]}"; do
                # Create a unique prefix for this model and run
                prefix="${m}_run$run_id"
                
                echo "Running: $prefix ($((run_id + 1))/$total_runs) | model=$m, aug=$feat_augment | layer=$l, neighbor=$n"
                
                python src/05_static_graph_tuning.py \
                    --model "$m" \
                    --prefix "$prefix" \
                    --lr "$lr" \
                    --bs "$bs" \
                    --n_epoch "$epochs" \
                    --loss focal \
                    --node_dim "$node_dim" \
                    --n_layer "$l" \
                    --n_neighbor "$n" \
                    --early_stop_higher_better \
                    $augment_flag

                # Increment counter
                ((run_id++))
            done
        done
    done
done