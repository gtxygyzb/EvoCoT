pip install antlr4-python3-runtime==4.11.1
export CUDA_VISIBLE_DEVICES="0,1,2,3"

PROMPT_TYPE=""
MODEL_NAME_OR_PATH=""

cd evaluation
bash sh/eval_trainset.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH