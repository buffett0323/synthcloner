CUDA_VISIBLE_DEVICES=4 python eval_proposed_seen.py \
    --device cuda:0 \
    --bs 32 \
    --config configs/config_proposed_no_ca.yaml \
    --checkpoint /home/buffett/nas_data/EDM_FAC_LOG/0818_proposed_no_ca_new/ckpt/checkpoint_400000.pt \
    --output_dir /home/buffett/nas_data/EDM_FAC_LOG/final_eval/0818_proposed_no_ca_new/detail/400k_seen
