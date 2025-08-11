import json
import requests
import os
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer
import re
import random

from verl.utils.reward_score.hf_math_verify import compute_score

import pdb
# 设置命令行参数解析
parser = argparse.ArgumentParser(description="Run model with specified parameters.")
parser.add_argument("--work_dir", type=str, required=True, help="Working directory")
parser.add_argument("--model_name", type=str, required=True, help="Model name")
parser.add_argument("--model_path", type=str, required=True, help="Model path")

parser.add_argument("--start", type=int, default=1, help="Start index for processing")
parser.add_argument("--end", type=int, default=1, help="End index for processing")
parser.add_argument("--step", type=int, default=1, help="Number of steps to delete from the end")

args = parser.parse_args()

work_dir = args.work_dir
model_name = args.model_name
model_path = args.model_path
start = args.start
end = args.end
step = args.step

save_dir = os.path.join(work_dir, model_name)
result_file = os.path.join(save_dir, "results.jsonl")

def truncate(text: str) -> str:
    matches = re.findall(r'\\boxed{(.*?)}', text)
    return matches[-1] if matches else None

def delete_step(text: str, step: int) -> str:
    # 从后往前，删除step个\n\n
    parts = text.split("\n\n")
    if len(parts) <= step:
        print(text)
        print(f"Not enough steps to delete {step} steps")   
        # pdb.set_trace()
        # raise ValueError("Not enough steps to delete")
        return None
    return "\n\n".join(parts[:-step])

final_data = []

for s in range(start, end + 1, step):
    mx = 0
    with open(result_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            mx = max(mx, int(data["idx"]))
            if data["idx"] < mx:
                break
            completions = data["completions"]
            if data["idx"] in [108, 167, 228]:
                continue

            random.shuffle(completions)

            for c in completions:
                reward = compute_score(
                    data_source=None,
                    solution_str=c,
                    ground_truth=data["gt"],
                    extra_info=None
                )
                # if data["gt"] == "335{,}670":
                #     print(c)
                #     print(len(completions))
                if reward < 0.5:
                    continue
                c = delete_step(c, s)
                if c is None:
                    continue
                # 构造 user prompt
                if "Qwen" or "Deep" in model_name:
                    user_prompt = {
                        "role": "user",
                        "content": data["question"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."
                    }
                else:
                    user_prompt = {
                        "role": "user",
                        "content": data["question"]
                    }
                
                # 构造完整 entry
                entry = {
                    "data_source": "custom/gsm8k",
                    "prompt": [
                        # {"role": "system", "content": "You are a helpful assistant."},
                        user_prompt,
                        {"role": "assistant", "content": c}
                    ],
                    "ability": "math",
                    "reward_model": {
                        "ground_truth": data["gt"],
                        "style": "rule"
                    },
                    "extra_info": {
                        "answer": data["answer"],
                        "index": data["idx"],
                        "question": data["question"],
                        "split": "train"
                    }
                }

                final_data.append(entry)
                break

import pandas as pd
df = pd.DataFrame(final_data)
parquet_file = os.path.join(save_dir, "step_train.parquet")

df.to_parquet(parquet_file, index=False)
print(f"Data saved to {parquet_file}")