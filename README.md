## Setup

### Install Enviroment

```
pip install -r requirements.txt
```

### 1. Generation Training Dataset

```
sh scripts/generate.sh
```

## 2. Combine  Generation Data

You only need to execute it when using the `generation.py` script.

```jsx
python gsil/combine.py --data_dir /path/of/your/iter_n_data
```
The final data will be stored in the `train_data` folder under /path/of/your/iter_n_data

## 3. Training

First, you need prepare one deepspeed profile `mae_hostfile` , which content like:

**Multi Host, Multi GPU**
```jsx
11.220.29.224 slots=8
11.220.39.208 slots=8
11.216.47.205 slots=8
11.216.61.163 slots=8
```

**Single Host, Multi GPU**
```jsx
11.220.29.224 slots=8
```

Second, you can use the following scripts to train the model.

```jsx
sh scripts/fintune.sh
```

```jsx
# base model path
model_path=alignment-handbook/zephyr-7b-sft-full
# training data dir path by combine.py
data_path=/path/zephyr_full_62k_greedy/iter0/train_data
# model save path
output_dir=/path/outputs_models/zephyr_full_62k_greedy_bernoulli_ao01s2
log_file=logs/zephyr_full_62k_greedy_bernoulli_ao01s2.log

if [ ! -d ${output_dir} ];then
    mkdir ${output_dir}
fi

nohup deepspeed --hostfile=/tmp/mae_hostfile gsil/run_gsil.py \
    --model_name_or_path ${model_path}\
    --torch_dtype "bfloat16" \
    --use_flash_attention_2 True \
    --dataset_path ${data_path} \
    --dataset_weight 1.0 \
    --dataset_splits "train" \
    --preprocessing_num_workers 12 \
    --bf16 True \
    --ddp_timeout 5400 \
    --loss_type "bernoulli_scale_shift"\
    --alpha 0.01 \
    --beta 2 \
    --do_eval False \
    --evaluation_strategy "no" \
    --hub_model_id "zephyr-7b-sft-full" \
    --learning_rate 5.0e-7 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --log_level "info" \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --lr_scheduler_type "linear"\
    --max_length 1024 \
    --max_prompt_length 512 \
    --num_train_epochs 1 \
    --optim rmsprop \
    --output_dir ${output_dir} \
    --deepspeed configs/deepspeed_config_bf16.json \
    --push_to_hub False \
    --save_strategy "epoch" \
    --save_total_limit 6 \
    --seed 42 \
    --warmup_steps 30 \
    --warmup_ratio 0.1 > ${log_file}  2>&1 &

```

## 4. Evaluation

For our evaluation on the Open LLM Leaderboard, please use the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/b281b0921b636bc36ad05c0b0b0763bd6dd43463) repository at v0.3.1,
which is consistent with open_llm_leaderboard. Also, note that we set the number of few shot examples to be the same as instructed on the [Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard).

Humaneval：https://github.com/OpenBMB/Eurus?tab=readme-ov-file

Mt-bench：https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge