set -ex

PROMPT_TYPE=$1
MODEL_NAME_OR_PATH=$2
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR=${TIMESTAMP}/${MODEL_NAME_OR_PATH}/math_eval

SPLIT="test"
NUM_TEST_SAMPLE=-1

# English open datasets
#DATA_NAME="gsm8k,math,svamp,asdiv,mawps,carp_en,tabmwp,minerva_math,gaokao2023en,olympiadbench,college_math"
#DATA_NAME="gsm8k,math,aime24,amc23,minerva_math,olympiadbench"
DATA_NAME="aime24,amc23,minerva_math,olympiadbench"

TOKENIZERS_PARALLELISM=false \
python3 -u math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --temperature 0.6 \
    --n_sampling 8 \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite \

echo "Evaluation completed. Results saved to ${OUTPUT_DIR}."