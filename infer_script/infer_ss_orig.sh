python infer_ss_orig.py \
    --checkpoint /mnt/gestalt/home/buffett/EDM_FAC_LOG/0707_ss/ckpt/checkpoint_latest.pt \
    --config configs/config_ss_orig.yaml \
    --orig_audio /mnt/gestalt/home/buffett/EDM_FAC_NEW_DATA/rendered_ss_t_adsr_c_new/evaluation/T010_ADSR001_C010.wav \
    --ref_audio /mnt/gestalt/home/buffett/EDM_FAC_NEW_DATA/rendered_ss_t_adsr_c_new/evaluation/T000_ADSR000_C009.wav \
    --gt_audio /mnt/gestalt/home/buffett/EDM_FAC_NEW_DATA/rendered_ss_t_adsr_c_new/evaluation/T000_ADSR000_C010.wav \
    --output_dir ss_infer_long2short/ \
    --convert_type both \
    --device cuda:0 \
    --prefix "long2short_"
