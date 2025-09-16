CUDA_VISIBLE_DEVICES=2 python eval_proposed_detail_abl.py \
    --device cuda:0 \
    --bs 32 \
    --config configs/config_mn_ablation.yaml \
    --checkpoint /home/buffett/nas_data/EDM_FAC_LOG/0804_ablation/ckpt/checkpoint_800000.pt \
    --output_dir /home/buffett/nas_data/EDM_FAC_LOG/final_eval/0804_ablation/detail/400k
