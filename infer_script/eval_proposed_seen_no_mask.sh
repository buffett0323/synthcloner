CUDA_VISIBLE_DEVICES=4 python eval_proposed_seen.py \
    --device cuda:0 \
    --bs 32 \
    --config configs/config_proposed_no_mask.yaml \
    --checkpoint /home/buffett/nas_data/EDM_FAC_LOG/0817_proposed_no_mask_no_ca/ckpt/checkpoint_410000.pt \
    --output_dir /home/buffett/nas_data/EDM_FAC_LOG/final_eval/0817_proposed_no_mask_no_ca/detail/400k_seen
