CHECKPOINT=/home/buffett/nas_data/EDM_FAC_LOG/0730_mn_cross_attn_enc_v1_onset_only_mask_p0_5/ckpt/checkpoint_latest.pt
CONFIG=configs/config_mn_cross_attn_content_onset_only.yaml
OUTPUT_DIR_SHORT2LONG=0730_exp_cr_attn_short2long
OUTPUT_DIR_LONG2SHORT=0730_exp_cr_attn_long2short
DEVICE=cuda:2


python infer_adsr_short2long.py \
    --checkpoint $CHECKPOINT \
    --config $CONFIG \
    --orig_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T000_ADSR000_C000.wav \
    --ref_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T010_ADSR001_C028.wav \
    --gt_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T010_ADSR001_C000.wav \
    --output_dir $OUTPUT_DIR_SHORT2LONG/ \
    --convert_type both \
    --device $DEVICE \
    --prefix "both_"

python infer_adsr_short2long.py \
    --checkpoint $CHECKPOINT \
    --config $CONFIG \
    --orig_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T000_ADSR000_C000.wav \
    --ref_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T010_ADSR001_C028.wav \
    --gt_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T010_ADSR000_C000.wav \
    --output_dir $OUTPUT_DIR_SHORT2LONG/ \
    --convert_type timbre \
    --device $DEVICE \
    --prefix "timbre_"

python infer_adsr_short2long.py \
    --checkpoint $CHECKPOINT \
    --config $CONFIG \
    --orig_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T000_ADSR000_C000.wav \
    --ref_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T010_ADSR001_C028.wav \
    --gt_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T000_ADSR001_C000.wav \
    --output_dir $OUTPUT_DIR_SHORT2LONG/ \
    --convert_type adsr \
    --device $DEVICE \
    --prefix "adsr_"


python infer_adsr_short2long.py \
    --checkpoint $CHECKPOINT \
    --config $CONFIG \
    --orig_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T010_ADSR001_C028.wav \
    --ref_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T000_ADSR000_C000.wav \
    --gt_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T000_ADSR000_C028.wav \
    --output_dir $OUTPUT_DIR_LONG2SHORT/ \
    --convert_type both \
    --device $DEVICE \
    --prefix "both_"

python infer_adsr_short2long.py \
    --checkpoint $CHECKPOINT \
    --config $CONFIG \
    --orig_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T010_ADSR001_C028.wav \
    --ref_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T000_ADSR000_C000.wav \
    --gt_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T000_ADSR001_C028.wav \
    --output_dir $OUTPUT_DIR_LONG2SHORT/ \
    --convert_type timbre \
    --device $DEVICE \
    --prefix "timbre_"

python infer_adsr_short2long.py \
    --checkpoint $CHECKPOINT \
    --config $CONFIG \
    --orig_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T010_ADSR001_C028.wav \
    --ref_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T000_ADSR000_C000.wav \
    --gt_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T010_ADSR000_C028.wav \
    --output_dir $OUTPUT_DIR_LONG2SHORT/ \
    --convert_type adsr \
    --device $DEVICE \
    --prefix "adsr_"
