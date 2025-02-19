model=TinyLlama_v1.1
directory_path=outputs/train/TinyLlama_v1.1/2024-12-20-09-42-42/checkpoints
input_file=data/entities_50000/test.jsonl
for dirname in "$directory_path"/*/; do
model_name_or_path="$dirname"
    CUDA_VISIBLE_DEVICES=0 python predict.py \
        --model_name_or_path $model_name_or_path \
        --input_file $input_file \
        --max_new_tokens 15
done