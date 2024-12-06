#!/bin/bash

devices=(0 1 2 3)
device_ids=$(echo "${devices[@]}" | tr ' ' ',')
device_count=${#devices[@]}

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="${device_ids}" accelerate launch --num_machines=1 \
  --multi_gpu \
  --num_processes="${device_count}" \
  --gpu_ids="${device_ids}" \
  --mixed_precision=fp16 \
  --dynamo_backend=no ../src/train/roberta.py \
  --output_dir <output_dir> \
  --training_set_path <training_set_path> \
  --validation_set_path <validation_set_path> \
  --base_model_name_or_path chinese_roberta_L-12_H-768 \
  --task_type <task_type> \
  --do_train True \
  --do_eval True \
  --evaluation_strategy epoch \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 1 \
  --learning_rate 6e-5 \
  --num_train_epochs 30 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --save_strategy epoch \
  --save_total_limit 2 \
  --bf16 True \
  --tf32 True \
  --report_to none
