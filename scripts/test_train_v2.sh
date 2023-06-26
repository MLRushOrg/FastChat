deepspeed --num_gpus=1 /data/FastChat/fastchat/train/train.py \
    --model_name_or_path baichuan-inc/baichuan-7B \
    --data_path /data/FastChat/data/test.json\
    --fp16 True \
    --output_dir ./output_test \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 2 \
    --save_strategy "steps" \
    --save_steps 1200 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --deepspeed /data/FastChat/playground/deepspeed_config_s2.json



##
    #--lazy_preprocess True