import json
import os
import argparse

parser = argparse.ArgumentParser(description="Run model with specified parameters.")

parser.add_argument("--results_dir", type=str, required=True, help="Directory containing the results files")
parser.add_argument("--prompt_type", type=str, default="reasoning", help="Type of prompt to use")
args = parser.parse_args()
results_dir = args.results_dir
prompt_type = args.prompt_type

file_path = [
    os.path.join(results_dir, f"gsm8k/train_{prompt_type}_-1_seed0_t0.6_s0_e-1.jsonl"),
    os.path.join(results_dir, f"math/train_{prompt_type}_-1_seed0_t0.6_s0_e-1.jsonl"),
]

wrong_save_path = os.path.join(results_dir, "wrong_predictions.jsonl")

# Given a question and its final answer, generate a clear, detailed, and step-by-step reasoning process that connects the question to the answer. The reasoning should be thorough, logically sound, and clearly justified, explaining exactly how the answer is derived without skipping any important steps. Please reason step by step, and put your final answer within \\boxed{{}}.
# Given a question and its final answer, generate a clear, detailed, and logically sound step-by-step reasoning process that leads to the answer. Each step should be separated by a double newline "\\n\\n" to ensure clarity. Do not skip any important steps in the reasoning. Please reason step by step, and put your final answer within \\boxed{{}}.

PROMPT_TEMPLATES = """
Given a question and its final answer, generate a clear, detailed, and logically sound step-by-step reasoning process that leads to the answer.

Each step should be separated by two newline characters \\n\\n for clarity.

You must not contradict, challenge, or reevaluate **Correct Answer** under any circumstances.

Question: {QUESTION}

Correct Answer: {ANSWER}

Now write the reasoning. Ensure that your reasoning matches the correct answer.
""".strip()

with open(wrong_save_path, "w") as wrong_file:
    for file in file_path:
        with open(file, "r") as f:
            for line in f:
                data = json.loads(line)
                score = data["score"]
                if True not in score:
                    # print("Wrong prediction found!")
                    # print(data)
                    # save to wrong_file
                    data["prompt"] = PROMPT_TEMPLATES.format(
                        QUESTION=data["question"],
                        ANSWER=data["gt"]
                    )
                    wrong_file.write(json.dumps(data) + "\n")
        