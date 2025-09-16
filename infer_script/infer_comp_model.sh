python infer_comp_model.py \
    --checkpoint /home/buffett/nas_data/EDM_FAC_LOG/0725_mn_cross_attn_enc_v1/ckpt/checkpoint_latest.pt \
    --config configs/config_mn_cross_attn_enc_v1.yaml \
    --orig_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T000_ADSR000_C000.wav \
    --ref_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T010_ADSR001_C028.wav \
    --gt_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T010_ADSR001_C000.wav \
    --output_dir 0725_exp_enc_v1_comp/ \
    --convert_type both \
    --device cuda:2 \
    --prefix "short2long_"


python infer_comp_model.py \
    --checkpoint /home/buffett/nas_data/EDM_FAC_LOG/0725_mn_cross_attn_enc_v1/ckpt/checkpoint_latest.pt \
    --config configs/config_mn_cross_attn_enc_v1.yaml \
    --orig_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T010_ADSR001_C028.wav \
    --ref_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T000_ADSR000_C000.wav \
    --gt_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T000_ADSR000_C028.wav \
    --output_dir 0725_exp_enc_v1_comp/ \
    --convert_type both \
    --device cuda:2 \
    --prefix "long2short_"


python infer_comp_model.py \
    --checkpoint /home/buffett/nas_data/EDM_FAC_LOG/0725_mn_cross_attn_enc_v1_onset_only/ckpt/checkpoint_latest.pt \
    --config configs/config_mn_cross_attn_content_onset_only.yaml \
    --orig_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T000_ADSR000_C000.wav \
    --ref_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T010_ADSR001_C028.wav \
    --gt_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T010_ADSR001_C000.wav \
    --output_dir 0725_exp_enc_v1_onset_only_comp/ \
    --convert_type both \
    --device cuda:2 \
    --prefix "short2long_"


python infer_comp_model.py \
    --checkpoint /home/buffett/nas_data/EDM_FAC_LOG/0725_mn_cross_attn_enc_v1_onset_only/ckpt/checkpoint_latest.pt \
    --config configs/config_mn_cross_attn_content_onset_only.yaml \
    --orig_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T010_ADSR001_C028.wav \
    --ref_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T000_ADSR000_C000.wav \
    --gt_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T000_ADSR000_C028.wav \
    --output_dir 0725_exp_enc_v1_onset_only_comp/ \
    --convert_type both \
    --device cuda:2 \
    --prefix "long2short_"
