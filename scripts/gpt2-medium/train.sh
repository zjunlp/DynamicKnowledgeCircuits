model=gpt2-medium
model_name_or_path=/mnt/8t/oyx/PLMs/${model}
train_file=data/entities_50000/train.jsonl
validation_file=data/entities_50000/validation.jsonl
CUDA_VISIBLE_DEVICES=0,1 python train.py \
    --model $model \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name $model_name_or_path \
    --train_file $train_file \
    --validation_file $validation_file \
    --load_data_from_cache True \
    --block_size 1024 \
    --output_dir outputs/train/$model/$(date +"%Y-%m-%d-%H-%M-%S")/checkpoints \
    --do_train \
    --do_eval \
    --eval_strategy steps \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-3 \
    --weight_decay 0.1 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-6 \
    --num_train_epochs 15 \
    --lr_scheduler_type constant \
    --logging_dir outputs/train/$model/$(date +"%Y-%m-%d-%H-%M-%S") \
    --logging_strategy steps \
    --logging_steps 50 \
    --save_strategy epoch \
    --report_to wandb