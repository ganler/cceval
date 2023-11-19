set -eux

export PYTHONPATH=$PWD
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/scratch/huggingface

export lang=python
export dtype=bf16
export ts_lib=./build/$lang-lang-parser.so

export model_name=deepseek-ai/deepseek-coder-6.7b-base

# depth1
export prompt_file=./data/crosscodeeval_data/$lang/line_completion_depth1_filtered.jsonl # x
export model_type=codelm_cfc # x codelm_cfc | codelm
export output_dir=$HOME/cceval_results_dscoder/depth1 # x

max_seq_lengths=(2 4 8) # in "k"
batch_sizes=(24 12 6)

for ((i=0; i<${#max_seq_lengths[@]}; i++)); do
    max_seq_length=$((max_seq_lengths[i] * 1024))
    cfc_seq_length=$((max_seq_length - 1024))
    batch_size=${batch_sizes[i]}

    accelerate launch eval.py \
            --model_type $model_type \
            --model_name_or_path $model_name \
            --cfc_seq_length $cfc_seq_length \
            --prompt_file $prompt_file \
            --gen_length 50 \
            --max_seq_length $max_seq_length \
            --batch_size $batch_size \
            --output_dir ${output_dir}.${max_seq_lengths[i]}k \
            --dtype $dtype \
            --num_return_sequences 1 \
            --overwrite_cache True \
            --ts_lib $ts_lib \
            --language $lang \
            --temperature 0
done