#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8

home_dir=`realpath ../..`
export PYTHONPATH=${PYTHONPATH}:${home_dir}

in_file=$1
retriever=$2
granularity=$3
index_expansion=${4:-"none"}
index_expansion_join_mode=${5:-"none"}   # separate split-separate merge replace
index_expansion_cache=${6:-"none"}
aux_model_alias=${7:-"none"}
outfile_prefix=${8:-"none"}

declare -A model_zoo
model_zoo["none"]="none"
model_zoo["llama-3.1-8b-instruct"]="meta-llama/Meta-Llama-3.1-8B-Instruct"
model_zoo["llama-3.1-8b-instruct-ICL"]="meta-llama/Meta-Llama-3.1-8B-Instruct"
model_zoo["llama-3.1-70b-instruct"]="meta-llama/Meta-Llama-3.1-70B-Instruct"
aux_model=${model_zoo["$aux_model_alias"]}

export HF_HOME=${home_dir}/model_cache/
cache_dir=${home_dir}/model_cache/

if [[ $index_expansion == "none" ]]; then
       out_dir=${home_dir}/retrieval_logs/${retriever}/${granularity}/
elif [[ $index_expansion == "session-summ" || $index_expansion == "session-keyphrase" || $index_expansion == "session-userfact" || $index_expansion == "turn-keyphrase" || $index_expansion == "turn-userfact" ]]; then
       out_dir=${home_dir}/retrieval_logs/${retriever}_expansion_w_${index_expansion}/${aux_model_alias}/joinmode${index_expansion_join_mode}/
else
       echo "Unrecognized index expansion method"
       exit 1
fi
mkdir -p $out_dir

python3 run_retrieval.py \
       --in_file $in_file \
       --retriever $retriever \
       --granularity $granularity \
       --index_expansion_method ${index_expansion} \
       --index_expansion_result_join_mode ${index_expansion_join_mode} \
       --index_expansion_llm ${aux_model} \
       --index_expansion_result_cache ${index_expansion_cache} \
       --out_dir ${out_dir} \
       --outfile_prefix ${outfile_prefix} \
       --cache_dir ${cache_dir}
       
