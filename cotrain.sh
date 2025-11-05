tpu watch v6 -n 32 -- pi_combined_cot_v6 \
    --exp-name=oxe_rotation_smalllr_predloss_cotrain --fsdp-devices=32 --batch-size=2048 \
    --data.no-use-json-actions --data.shuffle-buffer-size=400000 --resume --model.max-token-len=220 \
    --model.enable-prediction-training --model.no-enable-action-training --model.enable-langact-training \
    --model.prompt-format=pi05 --data.language_action_format_name=with_rotation \
    --lr-schedule.peak-lr=0.00005 --lr-schedule.decay-lr=0.00005 \
    --data.data-mix=oxe_magic_soup_vqa --model.enable-vqa-training --model.prediction_loss_weight=0.2 \
    --weight-loader.kind=paligemma2 --weight-loader.params-path=gs://v6_east1d/cache/paligemma2-3b-pt-224.b16.npz --model.paligemma_variant=gemma2_2b --model.action_expert_variant=gemma2_300m