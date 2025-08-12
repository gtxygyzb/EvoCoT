# **EvoCoT**: Overcoming the Exploration Bottleneck in Reinforcement Learning

[![arXiv](https://img.shields.io/badge/arXiv-2405.16368-b31b1b.svg)](https://arxiv.org/abs/2508.07809)

**EvoCoT** is a self-**Evo**lving curriculum learning framework for LLM reasoning, built on **two-stage Chain-of-Thought (CoT) optimization**.

- **Stage 1:** The model generates its own CoT explanations from problems and final answers, which are filtered and verified to form step-by-step reasoning trajectories.  
- **Stage 2:** EvoCoT performs curriculum learning by progressively removing thinking steps to improve reasoning efficiency while constraining the exploration space.

This framework enables LLMs to enhance their reasoning capability with a controlled and self-improving workflow.

# 🧱 Installation

We use verl docker in `requirements.txt`:

```bash
docker pull whatcanyousee/verl:ngc-cu124-vllm0.8.4-sglang0.4.5-mcore0.12.0-te2.2
```

# 🛠️ Usage Guide

## 1. Unsolved Problems Collection

Run the following script:

```bash
sh ./scripts/1error_test.sh
````

Edit the following variables in the script to configure `PROMPT_TYPE` and `MODEL_NAME_OR_PATH`:

```bash
PROMPT_TYPE=""
MODEL_NAME_OR_PATH=""
```

## 2. Answer-Guided Reasoning Path Self-Generation

Run the following script:

```bash
sh ./scripts/2QA2reasoning.sh
````

Edit the following variables in the script to configure model paths and prompts:

```bash
model_path=""
model_name=
lora_path=""
prompt_file=""
```

**Note:** `prompt_file` should be the result saved from **Step 1**.

## 3. Step-Wise Curriculum Learning

Run the following script:

```bash
sh ./scripts/3grpo.sh
````

Edit the following variables in the script to configure the model:

```bash
MODEL_NAME=""
MODEL_PATH=""
```

**Note:**
`TRAIN_DATA="./Self-reasoning/${MODEL_NAME}/step_train.parquet"` should be the output generated from **Step 2**.

## 4. Evaluation

Run the following script:

```bash
sh ./scripts/4eval.sh
````

Edit the following variables in the script to configure evaluation:

```bash
PROMPT_TYPE=""
MODEL_NAME=""
```

# 🤝 Acknowledgements

This project reuses code from the following repositories:

* [verl](https://github.com/volcengine/verl)
* [Math-Verify](https://github.com/huggingface/Math-Verify)
* [vLLM](https://github.com/vllm-project/vllm)
* [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math)
* [trl](https://github.com/huggingface/trl)

# 📜 Citation

```
@misc{liu2025evocotovercomingexplorationbottleneck,
      title={EvoCoT: Overcoming the Exploration Bottleneck in Reinforcement Learning}, 
      author={Huanyu Liu and Jia Li and Chang Yu and Taozhi Chen and Yihong Dong and Lecheng Wang and Hu XiaoLong and Ge Li},
      year={2025},
      eprint={2508.07809},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.07809}, 
}
```

# 📄 License

This repository includes components licensed under the Apache License 2.0.
