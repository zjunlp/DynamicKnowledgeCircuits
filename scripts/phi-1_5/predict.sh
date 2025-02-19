model=phi-1_5
directory_path=outputs/train/phi-1_5/2025-02-06-00-04-57/checkpoints
input_file=data/entities_50000/test.jsonl
for dirname in "$directory_path"/checkpoint-*/; do
model_name_or_path="$dirname"
    CUDA_VISIBLE_DEVICES=0 python predict.py \
        --model_name_or_path $model_name_or_path \
        --input_file $input_file \
        --max_new_tokens 15
done