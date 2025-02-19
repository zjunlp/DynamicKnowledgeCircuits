model=gpt2
directory_path=outputs/train/gpt2/2024-11-08-15-22-31/checkpoints
input_file=data/entities_50000/test.jsonl
# model_name_or_path=outputs/train/gpt2/2024-11-08-15-22-31/checkpoints/checkpoint-8100
for dirname in "$directory_path"/*/; do
    model_name_or_path="$dirname"
    CUDA_VISIBLE_DEVICES=0 python batch_predict.py \
        --model_name_or_path $model_name_or_path \
        --input_file $input_file \
        --max_new_tokens 15 \
        --batch_size 5
done