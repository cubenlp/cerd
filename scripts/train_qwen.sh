#!/bin/bash

deepspeed --hostfile="" --include localhost:0 --master_port 12600 ../src/train/qwen.py \
  --deepspeed zero3.json \
  --output_dir <output_dir> \
  --training_set_path <training_set_path> \
  --validation_set_path <validation_set_path> \
  --base_model_name_or_path Qwen/Qwen1.5-7B-Chat \
  --use_lora True \
  --lora_r 32 \
  --lora_alpha 32 \
  --do_train True \
  --do_eval False \
  --evaluation_strategy no \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --per_device_eval_batch_size 1 \
  --model_max_length 1024 \
  --learning_rate 2e-4 \
  --warmup_ratio 0.03 \
  --num_train_epochs 1 \
  --max_steps 250 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps 0.2 \
  --save_total_limit 5 \
  --bf16 True \
  --tf32 True \
  --gradient_checkpointing False \
  --report_to none

