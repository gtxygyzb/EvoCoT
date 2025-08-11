import json
import os
import argparse
import random
from tqdm import tqdm
from verl.utils.reward_score.hf_math_verify import compute_score

# 设置命令行参数解析
parser = argparse.ArgumentParser(description="Run model with specified parameters.")
parser.add_argument("--work_dir", type=str, required=True, help="Working directory")
parser.add_argument("--model_name", type=str, required=True, help="Model name")
parser.add_argument("--model_path", type=str, required=True, help="Model path")

args = parser.parse_args()

work_dir = args.work_dir
model_name = args.model_name
model_path = args.model_path

save_dir = os.path.join(work_dir, model_name)
result_file = os.path.join(save_dir, "results.jsonl")

final_data = []
mx = 0
with open(result_file, "r") as f:
    lines = f.readlines()
    for line in lines:
        data = json.loads(line)
        if data["idx"] in [108, 167, 228]:
            continue
        mx = max(mx, int(data["idx"]))
        completions = data["completions"]
        random.shuffle(completions)

        for c in completions:
            reward = compute_score(
                data_source=None,
                solution_str=c,
                ground_truth=data["gt"],
                extra_info=None
            )
            if reward < 0.5:
                continue

            # 构造 user prompt
            if "Qwen" in model_name or "Deep" in model_name:
                user_prompt = {
                    "role": "user",
                    "content": data["question"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."
                }
            else:
                user_prompt = {
                    "role": "user",
                    "content": data["question"]
                }

            entry = {
                "data_source": "custom/gsm8k",
                "prompt": [
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
            break  # 只保留一个满足条件的 completion

# 保存为 JSON 文件
output_path = os.path.join(save_dir, f"sft_train_{model_name}.json")
with open(output_path, "w") as f:
    json.dump(final_data, f, indent=2)

print(f"Data saved to {output_path}")
