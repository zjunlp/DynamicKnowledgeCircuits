model=gpt2
directory_path=outputs/train/gpt2-medium/2024-12-19-18-08-58/checkpoints
input_file=data/entities_50000/test.jsonl
for dirname in "$directory_path"/*/; do
model_name_or_path="$dirname"
    CUDA_VISIBLE_DEVICES=1 python predict.py \
        --model_name_or_path $model_name_or_path \
        --input_file $input_file \
        --max_new_tokens 15
done