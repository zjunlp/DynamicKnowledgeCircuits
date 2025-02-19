model=TinyLlama_v1.1
directory_path=outputs/train/TinyLlama_v1.1/2024-12-20-09-42-42/checkpoints
circuit_n=300

test_data_file=data/entities_50000/new_test.jsonl

for task in "city" "company" "major"; do
    eval_data_file=data/entities_50000/circuit_${circuit_n}/${model}/${task}.jsonl
    for dirname in "$directory_path"/checkpoint-*/; do
        model_path="$dirname"
        for source_type in "new" "revised"; do
            for source_frequency in "high" "medium" "low"; do
                target_type=$source_type
                target_frequency=$source_frequency
                CUDA_VISIBLE_DEVICES=0 python circuit_eval.py \
                    --model $model \
                    --model_path $model_path \
                    --task $task \
                    --eval_data_file $eval_data_file \
                    --test_data_file $test_data_file \
                    --source_type $source_type \
                    --source_frequency $source_frequency \
                    --target_type $target_type \
                    --target_frequency $target_frequency \
                    --batch_size 16 \
                    --method "EAP-IG" \
                    --topn 50000
            done
        done
    done
done
