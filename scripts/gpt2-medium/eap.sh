model=gpt2-medium
directory_path=outputs/train/gpt2-medium/2024-12-19-18-08-58/checkpoints
circuit_n=300

for task in "city" "company" "major"; do
    data_file=data/entities_50000/circuit_${circuit_n}/${model}/${task}.jsonl
    for dirname in "$directory_path"/checkpoint-*/; do
        model_path="$dirname"
        for type in "new" "revised"; do
            for frequency in "high" "medium" "low"; do
                CUDA_VISIBLE_DEVICES=0 python circuit_discovery.py \
                    --model $model \
                    --model_path $model_path \
                    --task $task \
                    --data_file $data_file \
                    --type $type \
                    --frequency $frequency \
                    --batch_size 64 \
                    --method "EAP-IG"
            done
        done
    done
done
