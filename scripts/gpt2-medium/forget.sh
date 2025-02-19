model=gpt2-medium
model_name_or_path=outputs/train/gpt2-medium/2024-12-19-18-08-58/checkpoints
train_file=data/forget_new_entities_50000/train.jsonl
validation_file=data/entities_50000/validation.jsonl
replay_file=data/forget_original_entities_50000/train.jsonl
for replay_ratio in 0.1 0.2 1
do
    CUDA_VISIBLE_DEVICES=0 python forget.py \
        --model $model \
        --model_name_or_path $model_name_or_path \
        --tokenizer_name $model_name_or_path \
        --train_file $train_file \
        --validation_file $validation_file \
        --load_data_from_cache False \
        --block_size 1024 \
        --output_dir outputs/forget/$model/$(date +"%Y-%m-%d-%H-%M-%S")/checkpoints \
        --do_train \
        --do_eval \
        --eval_strategy steps \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 8 \
        --learning_rate 1e-4 \
        --weight_decay 0.1 \
        --adam_beta1 0.9 \
        --adam_beta2 0.95 \
        --adam_epsilon 1e-6 \
        --num_train_epochs 5 \
        --lr_scheduler_type constant \
        --logging_dir outputs/forget/$model/$(date +"%Y-%m-%d-%H-%M-%S") \
        --logging_strategy steps \
        --logging_steps 5 \
        --save_strategy steps \
        --save_steps 1 \
        --report_to wandb \
        --do_replay True \
        --replay_file $replay_file \
        --replay_ratio $replay_ratio
done