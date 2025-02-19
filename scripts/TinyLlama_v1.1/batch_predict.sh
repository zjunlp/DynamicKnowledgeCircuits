model=TinyLlama_v1.1
directory_path=outputs/train/TinyLlama_v1.1/2024-10-21-20-45-58/checkpoints
input_file=data/entities_50000/test.jsonl
model_name_or_path=outputs/train/TinyLlama_v1.1/2024-10-21-20-45-58/checkpoints/checkpoint-4002
# for dirname in "$directory_path"/*/; do
# model_name_or_path="$dirname"
CUDA_VISIBLE_DEVICES=1 python batch_predict.py \
    --model_name_or_path $model_name_or_path \
    --input_file $input_file \
    --max_new_tokens 15 \
    --batch_size 5
# done