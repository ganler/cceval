set -eux

export PYTHONPATH=$PWD
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/scratch/huggingface

export lang=python
export dtype=bf16
export ts_lib=./build/$lang-lang-parser.so

export cfc_seq_length=7168 # 7k cfc ctx
export max_seq_length=8192 # 8k max ctx
export batch_size=6
export model_name=deepseek-ai/deepseek-coder-6.7b-base

# caag
# export model_type=codelm_cfc # x codelm_cfc | codelm
# export prompt_file=./data/crosscodeeval_data/$lang/line_completion_caag_filtered.jsonl # x
# export output_dir=$HOME/cceval_results_dscoder/caag # x

# accelerate launch eval.py \
#         --model_type $model_type \
#         --model_name_or_path $model_name \
#         --cfc_seq_length $cfc_seq_length \
#         --prompt_file $prompt_file \
#         --gen_length 50 \
#         --max_seq_length $max_seq_length \
#         --batch_size $batch_size \
#         --output_dir $output_dir \
#         --dtype $dtype \
#         --num_return_sequences 1 \
#         --overwrite_cache True \
#         --ts_lib $ts_lib \
#         --language $lang \
#         --temperature 0

# caag-simplify-fn
export model_type=codelm_cfc # x codelm_cfc | codelm
export prompt_file=./data/crosscodeeval_data/$lang/line_completion_caag_simpfn_filtered.jsonl # x
export output_dir=$HOME/cceval_results_dscoder/caag_simpfn # x

accelerate launch eval.py \
        --model_type $model_type \
        --model_name_or_path $model_name \
        --cfc_seq_length $cfc_seq_length \
        --prompt_file $prompt_file \
        --gen_length 50 \
        --max_seq_length $max_seq_length \
        --batch_size $batch_size \
        --output_dir $output_dir \
        --dtype $dtype \
        --num_return_sequences 1 \
        --overwrite_cache True \
        --ts_lib $ts_lib \
        --language $lang \
        --temperature 0

# no xfile
# export model_type=codelm # or codelm for no cross-file context eval
# export prompt_file=./data/crosscodeeval_data/$lang/line_completion_filtered.jsonl
# export output_dir=$HOME/cceval_results_dscoder/no_xfile

# accelerate launch eval.py \
#         --model_type $model_type \
#         --model_name_or_path $model_name \
#         --cfc_seq_length $cfc_seq_length \
#         --prompt_file $prompt_file \
#         --gen_length 50 \
#         --max_seq_length $max_seq_length \
#         --batch_size $batch_size \
#         --output_dir $output_dir \
#         --dtype $dtype \
#         --num_return_sequences 1 \
#         --overwrite_cache True \
#         --ts_lib $ts_lib \
#         --language $lang \
#         --temperature 0


# repo coder
export model_type=codelm_cfc # x codelm_cfc | codelm
export prompt_file=./data/crosscodeeval_data/$lang/line_completion_rg1_bm25_filtered.jsonl # x
export output_dir=$HOME/cceval_results_dscoder/repocoder # x

accelerate launch eval.py \
        --model_type $model_type \
        --model_name_or_path $model_name \
        --cfc_seq_length $cfc_seq_length \
        --prompt_file $prompt_file \
        --gen_length 50 \
        --max_seq_length $max_seq_length \
        --batch_size $batch_size \
        --output_dir $output_dir \
        --dtype $dtype \
        --num_return_sequences 1 \
        --overwrite_cache True \
        --ts_lib $ts_lib \
        --language $lang \
        --temperature 0

# oracle: line_completion_oracle_bm25_filtered
export model_type=codelm_cfc # x codelm_cfc | codelm
export prompt_file=./data/crosscodeeval_data/$lang/line_completion_oracle_bm25_filtered.jsonl # x
export output_dir=$HOME/cceval_results_dscoder/oracle # x

accelerate launch eval.py \
        --model_type $model_type \
        --model_name_or_path $model_name \
        --cfc_seq_length $cfc_seq_length \
        --prompt_file $prompt_file \
        --gen_length 50 \
        --max_seq_length $max_seq_length \
        --batch_size $batch_size \
        --output_dir $output_dir \
        --dtype $dtype \
        --num_return_sequences 1 \
        --overwrite_cache True \
        --ts_lib $ts_lib \
        --language $lang \
        --temperature 0

# depth1: line_completion_depth1_filtered
export model_type=codelm_cfc # x codelm_cfc | codelm
export prompt_file=./data/crosscodeeval_data/$lang/line_completion_depth1_filtered.jsonl # x
export output_dir=$HOME/cceval_results_dscoder/depth1 # x

accelerate launch eval.py \
        --model_type $model_type \
        --model_name_or_path $model_name \
        --cfc_seq_length $cfc_seq_length \
        --prompt_file $prompt_file \
        --gen_length 50 \
        --max_seq_length $max_seq_length \
        --batch_size $batch_size \
        --output_dir $output_dir \
        --dtype $dtype \
        --num_return_sequences 1 \
        --overwrite_cache True \
        --ts_lib $ts_lib \
        --language $lang \
        --temperature 0
