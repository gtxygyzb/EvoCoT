#!/bin/bash

pip install antlr4-python3-runtime==4.11.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

export CUDA_VISIBLE_DEVICES="4,5,6,7"

PROMPT_TYPE=""
MODEL_NAME=""

MODEL_NAME_OR_PATH="./${MODEL_NAME}/merged_hf_model"

cd evaluation
bash sh/eval_final.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH