# python infer_conversion.py \
#     --checkpoint /home/buffett/dataset/EDM_FAC_LOG/0608/ckpt/checkpoint_latest.pt \
#     --config configs/config.yaml \
#     --device cuda:3 \
#     --input_dir /home/buffett/dataset/EDM_FAC_DATA/evaluation/ \
#     --output_dir sample_audio/ \
#     --midi_dir /home/buffett/dataset/EDM_FAC_DATA/single_note_midi/evaluation/midi/ \
#     --midi_list_path info/midi_names_mixed_evaluation.txt \
#     --timbre_list_path info/timbre_names_mixed.txt \
#     --mode batch_convert \
#     --amount 10 \

python infer_conversion.py \
    --checkpoint /home/buffett/dataset/EDM_FAC_LOG/0608/ckpt/checkpoint_655000.pt \
    --config configs/config.yaml \
    --device cuda:0 \
    --input_dir /home/buffett/dataset/EDM_FAC_DATA/rendered_audio_new/lead_out/ \
    --output_dir testing_audio/ \
    --midi_dir /home/buffett/dataset/EDM_FAC_DATA/single_note_midi/evaluation/midi/ \
    --midi_list_path info/midi_names_lead_out.txt \
    --timbre_list_path info/timbre_names_lead_out.txt \
    --mode batch_convert \
    --amount 10 \
