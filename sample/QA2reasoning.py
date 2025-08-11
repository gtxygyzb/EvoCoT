import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import os
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer

# 设置命令行参数解析
parser = argparse.ArgumentParser(description="Run model with specified parameters.")
parser.add_argument("--work_dir", type=str, required=True, help="Working directory")
parser.add_argument("--model_name", type=str, required=True, help="Model name")
parser.add_argument("--model_path", type=str, required=True, help="Model path")
parser.add_argument("--prompt_file", type=str, required=True, help="Path to the prompt file")
parser.add_argument("--vllm_url", type=str, required=True, help="VLLM service URL")
parser.add_argument("--temperature", type=float, default=0, help="Temperature for sampling (default: 0)")
parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate (default: 1)")

import time
st_time = time.time()
# 解析命令行参数
args = parser.parse_args()

# 打印解析结果（调试用）
print(f"Work Directory: {args.work_dir}")
print(f"Model Name: {args.model_name}")
print(f"Model Path: {args.model_path}")
print(f"Prompt File: {args.prompt_file}")
print(f"VLLM URL: {args.vllm_url}")
print(f"Number of samples: {args.num_samples}")

# 后续逻辑使用解析的参数
work_dir = args.work_dir
model_name = args.model_name
model_path = args.model_path
prompt_file = args.prompt_file
vllm_url = args.vllm_url
n_samples = args.num_samples
temperature = args.temperature

tokenizer = AutoTokenizer.from_pretrained(model_path)

stop_words = ["</s>", "<|im_end|>", "<|endoftext|>", "Human", "\n\n## Question", "\n### Problem", "You are"]

def generate(prompt, temperature, n_samples):
    data = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": 3500,
        "temperature": temperature,
        "top_p": 0.95,
        "n": n_samples,
        "stop": stop_words,
    }
    try:
        response = requests.post(vllm_url, json=data)
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                # 如果有多个采样结果，返回所有的结果，或者返回第一个结果
                return [choice.get('text', '') for choice in result['choices']]
            else:
                return ['']
        else:
            print(f"请求失败，状态码: {response.status_code}, 错误信息: {response.text}")
            return ['']
    except Exception as e:
        print(f"请求时发生异常: {e}")
        return ['']


save_dir = os.path.join(work_dir, model_name)
print(save_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

result_file = os.path.join(save_dir, "results.jsonl")
if os.path.exists(result_file):
    os.remove(result_file)
    

def process_example(example, temperature, n_samples):
    prompt = example["prompt"]
    messages = [
        {"role": "user", "content": prompt}
    ]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"Processing prompt: {formatted_prompt}")
    completions = generate(formatted_prompt, temperature, n_samples)
    example["completions"] = completions
    print(f"Generated completions: {completions[0]}")
    return example

except_list = []
with open(prompt_file, "r") as f:
    examples = [json.loads(line) for line in f.readlines()]
examples_to_process = examples

with open(result_file, "a") as ff:
    # 使用多线程并行处理
    with ThreadPoolExecutor(max_workers=64) as executor:
        futures = {
            executor.submit(process_example, example, temperature, n_samples): i 
            for i, example in enumerate(examples_to_process)
        }

        # 用来存储按原始顺序的结果
        results = [None] * len(examples_to_process)

        for future in tqdm(as_completed(futures), total=len(futures)):
            index = futures[future]  # 获取原始索引
            try:
                result = future.result()
                results[index] = result  # 保持原始顺序
            except Exception as e:
                print(f"处理 index '{index}' 时发生错误: {e}")

        # 将结果写入文件（保持顺序）
        for result in results:
            if result is not None:
                ff.write(json.dumps(result) + "\n")
                ff.flush()  # 确保及时写入文件


ed_time = time.time()
with open(result_file + ".txt", "w") as f:
    f.write(str(ed_time - st_time))

