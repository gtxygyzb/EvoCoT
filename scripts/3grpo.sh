#!/bin/bash

pip install antlr4-python3-runtime==4.9.3 -i https://pypi.tuna.tsinghua.edu.cn/simple

MODEL_NAME=""
MODEL_PATH=""

TRAIN_DATA="./Self-reasoning/${MODEL_NAME}/step_train.parquet"
VAL_DATA="./Evaluation_data/gsm8k/test.parquet"

WANDB_RUN_NAME="${MODEL_NAME}_grpo"

LOG_DIR="./${MODEL_NAME}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/training_log_${TIMESTAMP}.log"

mkdir -p ${LOG_DIR}

echo "LOG_FILE: ${LOG_FILE}"

SAVE_PATH="./${MODEL_NAME}/save_model"

REWARD_FUNC_PATH="./verl/verl/utils/reward_score/hf_math_verify.py"  


(python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    custom_reward_function.path=${REWARD_FUNC_PATH} \
    data.train_files=${TRAIN_DATA} \
    data.train_batch_size=16 \
    data.val_files=${VAL_DATA} \
    data.val_batch_size=4 \
    data.max_prompt_length=2500 \
    data.max_response_length=4000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='EvoCoT' \
    trainer.experiment_name=${WANDB_RUN_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    trainer.default_local_dir=${SAVE_PATH}) 2>&1 | tee -a ${LOG_FILE}

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "===== Error, EXIT_CODE: $EXIT_CODE =====" >> ${LOG_FILE}
    echo "===== Final 100 lines =====" >> ${LOG_FILE}
    tail -n 100 ${LOG_FILE} >> ${LOG_FILE}.error
fi

echo "===== Time: $(date) =====" >> ${LOG_FILE}
echo "===== GPU =====" >> ${LOG_FILE}
nvidia-smi >> ${LOG_FILE}

echo "===== Process =====" >> ${LOG_FILE}
ps aux | grep python >> ${LOG_FILE}

echo "Done! Log save to: ${LOG_FILE}"


echo "Merge model..."
iteration=$(cat ./${MODEL_NAME}/save_model/latest_checkpointed_iteration.txt)
echo "Iteration: $iteration"
python verl/scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir ./${MODEL_NAME}/save_model/global_step_${iteration}/actor \
    --target_dir ./${MODEL_NAME}/merged_hf_model
