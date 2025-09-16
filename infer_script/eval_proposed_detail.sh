CUDA_VISIBLE_DEVICES=3 python eval_proposed_detail.py \
    --device cuda:0 \
    --bs 32 \
    --config configs/config_proposed.yaml \
    --checkpoint /home/buffett/nas_data/EDM_FAC_LOG/0804_proposed/ckpt/checkpoint_400000.pt \
    --output_dir /home/buffett/nas_data/EDM_FAC_LOG/final_eval/0804_proposed/detail/400k
