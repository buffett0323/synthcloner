CHECKPOINT=/home/buffett/nas_data/EDM_FAC_LOG/0730_mn_cross_attn_enc_v1_onset_only_mask_p0_5/ckpt/checkpoint_latest.pt
CONFIG=configs/config_mn_cross_attn_content_onset_only.yaml
OUTPUT_DIR=0730_ood_test
DEVICE=cuda:2


python infer_adsr_short2long.py \
    --checkpoint $CHECKPOINT \
    --config $CONFIG \
    --orig_audio /home/buffett/nas_data/EDM_FAC_MEDIUM_DATA/rendered_mn_t_adsr_c/train/T245_ADSR078_C093.wav \
    --ref_audio /home/buffett/nas_data/EDM_FAC_MEDIUM_DATA/rendered_mn_t_adsr_c/train/T098_ADSR037_C027.wav \
    --gt_audio /home/buffett/nas_data/EDM_FAC_MEDIUM_DATA/rendered_mn_t_adsr_c/train/T098_ADSR037_C093.wav \
    --output_dir $OUTPUT_DIR/ \
    --convert_type both \
    --device $DEVICE \
    --prefix "ood1_"


python infer_adsr_short2long.py \
    --checkpoint $CHECKPOINT \
    --config $CONFIG \
    --orig_audio /home/buffett/nas_data/EDM_FAC_MEDIUM_DATA/rendered_mn_t_adsr_c/train/T098_ADSR037_C027.wav \
    --ref_audio /home/buffett/nas_data/EDM_FAC_MEDIUM_DATA/rendered_mn_t_adsr_c/train/T245_ADSR078_C093.wav \
    --gt_audio /home/buffett/nas_data/EDM_FAC_MEDIUM_DATA/rendered_mn_t_adsr_c/train/T245_ADSR078_C027.wav \
    --output_dir $OUTPUT_DIR/ \
    --convert_type both \
    --device $DEVICE \
    --prefix "ood2_"
