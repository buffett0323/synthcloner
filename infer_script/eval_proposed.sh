#!/bin/bash

# Loop through checkpoints from 100000 to 400000 in steps of 10000
for checkpoint in $(seq 100000 10000 400000); do
    echo "Evaluating checkpoint: ${checkpoint}"

    python eval_proposed.py \
        --device cuda:4 \
        --bs 16 \
        --config configs/config_proposed.yaml \
        --checkpoint /home/buffett/nas_data/EDM_FAC_LOG/0804_proposed/ckpt/checkpoint_${checkpoint}.pt \
        --output_dir /home/buffett/nas_data/EDM_FAC_LOG/final_eval/0804_proposed/checkpoint_${checkpoint}
        # --save_samples

    echo "Completed evaluation for checkpoint: ${checkpoint}"
    echo "----------------------------------------"
done
