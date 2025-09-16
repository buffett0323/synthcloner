python infer_adsr_short2long.py \
    --checkpoint /home/buffett/nas_data/EDM_FAC_LOG/0723_mn_cross_attn_direct2dac/ckpt/checkpoint_latest.pt \
    --config configs/config_mn_cross_attn.yaml \
    --orig_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T000_ADSR000_C000.wav \
    --ref_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T010_ADSR001_C028.wav \
    --gt_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T010_ADSR001_C000.wav \
    --output_dir 0723_exp_direct2dac_short2long/ \
    --convert_type both \
    --device cuda:2 \
    --prefix "both_"

python infer_adsr_short2long.py \
    --checkpoint /home/buffett/nas_data/EDM_FAC_LOG/0723_mn_cross_attn_direct2dac/ckpt/checkpoint_latest.pt \
    --config configs/config_mn_cross_attn.yaml \
    --orig_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T000_ADSR000_C000.wav \
    --ref_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T010_ADSR001_C028.wav \
    --gt_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T010_ADSR000_C000.wav \
    --output_dir 0723_exp_direct2dac_short2long/ \
    --convert_type timbre \
    --device cuda:2 \
    --prefix "timbre_"

python infer_adsr_short2long.py \
    --checkpoint /home/buffett/nas_data/EDM_FAC_LOG/0723_mn_cross_attn_direct2dac/ckpt/checkpoint_latest.pt \
    --config configs/config_mn_cross_attn.yaml \
    --orig_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T000_ADSR000_C000.wav \
    --ref_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T010_ADSR001_C028.wav \
    --gt_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T000_ADSR001_C000.wav \
    --output_dir 0723_exp_direct2dac_short2long/ \
    --convert_type adsr \
    --device cuda:2 \
    --prefix "adsr_"


python infer_adsr_short2long.py \
    --checkpoint /home/buffett/nas_data/EDM_FAC_LOG/0723_mn_cross_attn_direct2dac/ckpt/checkpoint_latest.pt \
    --config configs/config_mn_cross_attn.yaml \
    --orig_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T010_ADSR001_C028.wav \
    --ref_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T000_ADSR000_C000.wav \
    --gt_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T000_ADSR000_C028.wav \
    --output_dir 0723_exp_direct2dac_long2short/ \
    --convert_type both \
    --device cuda:2 \
    --prefix "both_"

python infer_adsr_short2long.py \
    --checkpoint /home/buffett/nas_data/EDM_FAC_LOG/0723_mn_cross_attn_direct2dac/ckpt/checkpoint_latest.pt \
    --config configs/config_mn_cross_attn.yaml \
    --orig_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T010_ADSR001_C028.wav \
    --ref_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T000_ADSR000_C000.wav \
    --gt_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T000_ADSR001_C028.wav \
    --output_dir 0723_exp_direct2dac_long2short/ \
    --convert_type timbre \
    --device cuda:2 \
    --prefix "timbre_"

python infer_adsr_short2long.py \
    --checkpoint /home/buffett/nas_data/EDM_FAC_LOG/0723_mn_cross_attn_direct2dac/ckpt/checkpoint_latest.pt \
    --config configs/config_mn_cross_attn.yaml \
    --orig_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T010_ADSR001_C028.wav \
    --ref_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T000_ADSR000_C000.wav \
    --gt_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T010_ADSR000_C028.wav \
    --output_dir 0723_exp_direct2dac_long2short/ \
    --convert_type adsr \
    --device cuda:2 \
    --prefix "adsr_"
