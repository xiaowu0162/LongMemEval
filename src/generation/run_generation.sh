#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8

export home_dir=`realpath ../..`
export HF_HOME=${home_dir}/model_cache/

in_file=$1
model_alias=$2
retriever_alias=$3
topk_context=${4:-50}
history_format=${5:-"json"}
useronly=${6:-"false"}
reading_method=${7:-"con"}
merge_key_expansion_into_value=${8:-"none"}
suffix=${9:-"none"}

declare -A model_zoo
model_zoo["gpt-4o"]="gpt-4o-2024-08-06"
model_zoo["gpt-4o-mini"]="gpt-4o-mini-2024-07-18"
model_zoo["llama-3-8b-instruct"]="meta-llama/Meta-Llama-3-8B-Instruct"
model_zoo["llama-3-70b-instruct"]="meta-llama/Meta-Llama-3-70B-Instruct"
model_zoo["llama-3.1-8b-instruct"]="meta-llama/Meta-Llama-3.1-8B-Instruct"
model_zoo["llama-3.1-70b-instruct"]="meta-llama/Meta-Llama-3.1-70B-Instruct"
model_zoo["mistral-7b-instruct-v0.2"]="mistralai/Mistral-7B-Instruct-v0.2"
model_zoo["mistral-7b-instruct-v0.3"]="mistralai/Mistral-7B-Instruct-v0.3"
model_zoo["film-7b"]="In2Training/FILM-7B"
model_zoo["phi-3-medium-128k-instruct"]="microsoft/Phi-3-medium-128k-instruct"
model_zoo["phi-3.5-mini-instruct"]="microsoft/Phi-3.5-mini-instruct"
model_zoo["phi-4"]="microsoft/phi-4"
model=${model_zoo["$model_alias"]}

if [[ $model_alias == "gpt-4o" || $model_alias == "gpt-4o-mini" ]]; then
    # will call openai server
    openai_base_url_flag=""
    openai_key="YOUR_API_KEY"
    openai_org_flag="--openai_organization YOUR_ORGANIZATION"   # uclanlp
else
    # will call an openai server emulator served by e.g., vllm locally
    openai_base_url_flag="--openai_base_url http://localhost:8001/v1"
    openai_key="EMPTY"
    openai_org_flag=""
fi

retriever_type=""
if [[ $retriever_alias == "flat-bm25-turn" || $retriever_alias == "flat-contriever-turn" || $retriever_alias == "flat-stella-turn" || $retriever_alias == "flat-gtr-turn" ]]; then
    retriever_type="flat-turn"
elif [[ $retriever_alias == "flat-bm25-session" || $retriever_alias == "flat-contriever-session" || $retriever_alias == "flat-stella-session" || $retriever_alias == "flat-gtr-session" ]]; then
    retriever_type="flat-session"
elif [[ $retriever_alias == "full-history-turn" ]]; then
    retriever_type="orig-turn"
elif [[ $retriever_alias == "full-history-session" ]]; then
    retriever_type="orig-session"
elif [[ $retriever_alias == "no-retrieval" ]]; then
    retriever_type="no-retrieval"
else
    echo "retriever alias not recognized"
    exit 1
fi


if [[ $reading_method == "direct" ]]; then
    reading_flags="--cot false"
elif [[ $reading_method == "con" ]]; then
    reading_flags="--cot true"
elif [[ $reading_method == "con-separate" ]]; then
    reading_flags="--cot true --con true"
else
    echo "reading method not recognized"
    exit 1
fi

out_dir=${home_dir}/generation_logs/${retriever_alias}/${model_alias}/${reading_method}/
mkdir -p $out_dir

python run_generation.py \
       --in_file ${in_file} \
       --out_dir ${out_dir} \
       --out_file_suffix ${suffix} \
       --model_name ${model} --model_alias ${model_alias} \
       --retriever_type ${retriever_type} \
       --merge_key_expansion_into_value ${merge_key_expansion_into_value} \
       --openai_key ${openai_key} ${openai_org_flag} ${openai_base_url_flag}  \
       --topk_context ${topk_context} \
       --history_format ${history_format} \
       --useronly ${useronly} \
       ${reading_flags}
