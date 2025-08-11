model_path=""
model_name=""
lora_name=$model_name
lora_path=""
prompt_file=""

work_dir="./Self-reasoning"
n_sample=8 # Cot sample number
temperature=1

tensor_parallel_size=4
port=8050 

docker_name="vllm_model_${model_name}"

# docker kill $docker_name || true
# docker rm $docker_name || true

#  Start Docker VLLM
docker run -d --gpus '"device=0,1,2,3"' --name ${docker_name} \
  -v /nfs100:/nfs100 \
  -v /models:/models \
  -p ${port}:8000 \
  --shm-size=8g \
  vllm \
  --model=${model_path} \
  --served-model-name=${model_name} \
  --tensor-parallel-size=${tensor_parallel_size} \
  --max-num-seqs=64 \
  --gpu-memory-utilization=0.90 \
  --max-model-len=4096
  # --enable-lora \
  # --lora-modules "${lora_name}=${lora_path}" \
  # --max_lora_rank=32
# 
sleep 300
echo "Docker ${docker_name} Start!"


# Step 3: fine IP and get vllm_url
container_ip=$(docker inspect -f '{{.NetworkSettings.Networks.bridge.IPAddress}}' ${docker_name})
if [[ -z "${container_ip}" ]]; then
  echo "Error find IP"
  exit 1
fi
vllm_url="http://${container_ip}:8000/v1/completions"
echo "vllm_url: ${vllm_url}"


docker_name="Reasoning_${model_name}"
docker run --rm -it --name ${docker_name} \
    -v /nfs100:/nfs100 \
    -v /dev/shm:/dev/shm \
    -v /models:/models \
    trl_env:0910 sh -c "python ./sample/QA2reasoning.py \
    --work_dir=${work_dir} \
    --model_name=${lora_name} \
    --model_path=${model_path} \
    --prompt_file=${prompt_file} \
    --num_samples=${n_sample} \
    --vllm_url=${vllm_url} \
    --temperature=${temperature}"


START=1
END=10
STEP=1

docker exec -it verl sh -c "python ./sample/prepare_step_training_data.py \
    --work_dir=${work_dir} \
    --model_name=${lora_name} \
    --model_path=${model_path} \
    --start=${START} \
    --end=${END} \
    --step=${STEP}"

echo "Done!"