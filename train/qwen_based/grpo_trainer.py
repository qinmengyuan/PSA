"""VLLM_ALLOW_LONG_MAX_MODEL_LEN=1  python -m torch.distributed.run --nproc_per_node=4 -m proactive_defence.train.qwen_based.grpo_trainer 2>&1 | tee -a output.txt"""
import sys
sys.path.append('/data/psa/x-teaming')
import pickle
import os
from agents.gpt_evaluator import GPTJudge
import sqlite3
from datetime import datetime
from collections import Counter
from datasets import load_dataset
from proactive_defence import proactiveDefencer
from proactive_defence.utils import load_config
from proactive_defence.utils.format_convert import messages_to_nested_list
import torch
from tqdm import tqdm
from trl import apply_chat_template
from transformers import AutoTokenizer
from filelock import FileLock
from FastChat.fastchat.llm_judge.common import load_questions
from FastChat.fastchat.model.model_adapter import get_conversation_template
from proactive_defence.prompts.defence_prompts_md import AGGREGATE_ANALYSIS_TEMPLATE_SYS_MD, POTENTIALLY_HARMFUL_GENERATION, PROMPT_HARM, PROMPT_SAFE, SAFE_USER_REQUEST_PREDICTION
from proactive_defence.utils.SafetyScorer import SafetyScorer
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig
import json
from json_repair import repair_json
import re
from typing import List, Dict
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

judge = GPTJudge(model_name="openai/gpt-oss-120b")

DEBUG = True

# Create a standalone logger
my_train_logger = logging.getLogger("ultra")
my_train_logger.setLevel(logging.DEBUG)  # only affects this logger

# Create a handler writing to the console
console_handler = logging.StreamHandler()
if DEBUG:
    console_handler.setLevel(logging.DEBUG)  # handle DEBUG and above only

# Customizable output format
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console_handler.setFormatter(formatter)

my_train_logger.addHandler(console_handler)

# Do not propagate to the root logger
my_train_logger.propagate = False
my_train_logger.debug("Standalone logger configuration test message.")



config = load_config()

TRAIN_SAFE = config["TRAIN_SAFE"]
TRAIN_RH = config["TRAIN_RH"]
model_name = config["model_name"]
per_device_train_batch_size = config["per_device_train_batch_size"]
gradient_accumulation_steps = config["gradient_accumulation_steps"]
lr = config["learning_rate"]
save_dir = config["cache_pickle_read_dir"]
per_device_eval_batch_size = config['per_device_eval_batch_size']
REWARD_WHITE=config["REWARD_WHITE"]
print(TRAIN_SAFE, TRAIN_RH)

if REWARD_WHITE:

    import requests






# Always load the tokenizer from the official / model repo
tokenizer = AutoTokenizer.from_pretrained(model_name)


def parse_chat_to_messages(text: str, model="4") -> List[Dict]:
    """
    Parse raw text into a messages list of the specified format.
    """

    # Match <|im_start|>role and <|im_end|> with regex
    pattern = re.compile(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>", re.DOTALL)
    matches = pattern.findall(text)

    messages = []

    for role, content in matches:
        content = content.strip()
        # Treat the content as a whole and wrap it in a text dict
        # Split further here if multi-part content is needed
        if model == "3":
            messages.append({
                "role": role,
                "content": content
            })
        elif model == "4":
            messages.append({
                "role": role,
                "content": [
                    {
                        "type": "text",
                        "text": content
                    }
                ]
            })

    return messages[1:]


def check_format_score(reply_text: str) -> float:
    """Parsing succeeds: 0.4

Fields present: up to 0.3 (3 fields, 0.1 each)

Fields well-typed and non-empty: up to 0.3 (3 fields, 0.1 each)

    Return range: [0.0, 1.0]; higher is better
    """

    score = 0.0

    try:
        parsed = json.loads(reply_text)
        if parsed is None:
            return 0.0
        score += 0.4
    except json.JSONDecodeError:
        try:
            parsed = repair_json(reply_text, return_objects=True)
            if isinstance(parsed, list):
                parsed = [i for i in parsed if isinstance(i, dict)]
            if parsed:
                parsed = parsed[0]
            if parsed is None:
                return 0.0

            # parsed = json.loads(repaired_text)
            score += 0.25

            if not isinstance(parsed, dict):

                print("Parsed JSON is not a dictionary")
                print(f"Parsed JSON: {parsed}")
                return score
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            print(f"Original text: {reply_text}")
            return 0.0

    required_keys = ["Purpose", "Thoughts", "Proactive_Task"] if TRAIN_SAFE else [
        "User_Context", "Potential_Harm", "Warning_Signs", "Predictions"]

    # Points for field presence
    exist_count = sum(1 for k in required_keys if k in parsed)
    score += 0.3 * (exist_count / (3 if TRAIN_SAFE else 4))

    # Points for well-typed, non-empty fields
    valid_count = 0
    for k in required_keys:
        if k in parsed:
            val = parsed[k]
            if k != "Purpose" and k != "Thoughts" and k != "User_Context":
                if (isinstance(val, str) and val.strip() != "") or val is None or isinstance(val, list):
                    valid_count += 1
            else:
                if isinstance(val, str) and val.strip() != "":
                    valid_count += 1

    score += 0.3 * (valid_count / 3)

    return min(1.0, score)


# dataset = load_dataset("trl-lib/tldr", split="test")
# dataset = prepare_datasaets("SafeDial_bench")

# train_dataset, eval_dataset = split_dataset(dataset, train_ratio=0.2)

# eval_dataset = eval_dataset[:30]








os.makedirs(save_dir, exist_ok=True)
train_path = os.path.join(save_dir, "train_dataset.pkl")
eval_path = os.path.join(save_dir, "eval_dataset.pkl")




with open(train_path, "rb") as f:
    train_dataset = pickle.load(f)

with open(eval_path, "rb") as f:
    eval_dataset = pickle.load(f)[:20]

print(f"Black train dataset size: {len(train_dataset)}")
print(f"Black eval dataset size: {len(eval_dataset)}")
# [x["prompt"] for x in eval_dataset]
print(f"Train dataset size: {len(train_dataset)}")
print(f"Eval dataset size: {len(eval_dataset)}")
scorer = SafetyScorer(
    api_base="http://localhost:8000/v1", model="openai/gpt-oss-120b")


def evaluate_harm_with_gptoss(messages, attacker_request):
    """
    Call GPT-OSS for attack-prediction evaluation.

    """
    if DEBUG:
        return 1
    return scorer.calculate_harmful_score(messages, attacker_request)


def evaluate_safe_with_gptoss(messages, attacker_request):
    """
    Call GPT-OSS for safety-prediction evaluation.

    """
    if DEBUG:
        return 1
    if score := scorer.calculate_safety_score(messages, attacker_request):
        return score
    else:
        return score


def ngram_repeat_check(text, n=2, threshold=100):
    """
    text: input string
    n: n-gram length
    threshold: return True if any n-gram occurs at least this many times
    """
    text = text.replace(" ", "")
    if len(text) < n:
        return False
    # Build character-level n-grams
    ngrams = [tuple(text[i:i+n]) for i in range(len(text)-n+1)]
    counts = Counter(ngrams)
    # Check whether any n-gram occurs at least threshold times
    repeated = [ng for ng, count in counts.items() if count >= threshold]
    return any(count >= threshold for count in counts.values())


def init_reward_database(db_path="reward_scores.db"):
    """Initialize the database table."""
    conn = sqlite3.connect(db_path, timeout=60)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS reward_scores (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        safe_score REAL,
        format_score REAL,
        final_score REAL,
        alpha REAL,
        deduction_reason TEXT
    )
    ''')

    conn.commit()
    conn.close()


def record_score(db_path, safe_score, format_score, final_score, alpha, deduction_reason=None):
    lock = FileLock(db_path + ".lock")
    with lock:
        conn = sqlite3.connect(db_path, timeout=300)
        cur = conn.cursor()
        cur.execute('''
        INSERT INTO reward_scores (timestamp, safe_score, format_score, final_score, alpha, deduction_reason)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            safe_score,
            format_score,
            final_score,
            alpha,
            deduction_reason
        ))
        conn.commit()
        conn.close()


def check_md_format_score(completion_msgs):
    keywords = ["# Purpose", "# Thoughts", "# Proactive_Tasks"]if TRAIN_SAFE else ["# User Context", "# Potential Harm",
                                                                                   "# Warning Signs", "# Predictions"]
    scores = {k: 1 if k in completion_msgs else 0 for k in keywords}
    total_score = sum(scores.values())/len(keywords)
    return total_score, scores


def reward_func(prompts, completions, alpha=0.7, db_path="reward_scores.db", **kwargs):
    """
    Reward function: score each (prompt, completion) pair with llama4guard for task and format scores.

    Args:
        prompts: List[str], input prompts
        completions: List[str], model-generated texts
        alpha: float, weight of the safety score
        db_path: str, database file path
        kwargs: other fields

    Returns:
        List[float]: list of reward scores
    """
    try:
        rewards = []
        deferred_white = {}  # sample index -> (score term, format_score, raw white score); backfilled after per-prompt group normalization

        if True or TRAIN_SAFE:
            attacker_requests = kwargs.get("attacker_requests", False)
            if not attacker_requests:
                
                raise NotImplementedError(
                    "TRAIN_SAFE mode requires the attacker_requests argument")
        else:
            attacker_requests = [None] * len(prompts)
        if TRAIN_RH:
            task_dicts = kwargs.get("mytask", False)
            if not task_dicts:
                print(kwargs)
                raise NotImplementedError(
                    "TRAIN_RH mode requires the mytask argument")
                    
        for prompt_msgs, completion_msgs, attacker_request, task_dict in zip(prompts, completions, attacker_requests, task_dicts if TRAIN_RH else [None]*len(prompts)):

            safe_score = None
            format_score = None
            final_score = None
            deduction_reason = None
            current_res = None
            if isinstance(completion_msgs, list) and isinstance(completion_msgs[0], dict) and (current_res := completion_msgs[0].get('content', False)):
                pass
            elif isinstance(completion_msgs, str):
                print("Warning: expected chat format, but completion_msgs is str, not list of dicts.",completion_msgs)
                raise NotImplementedError("Only chat-format training is supported")
                # current_res=completion_msgs
            if TRAIN_RH:
                if not current_res:
                    print("-------TRAIN_RH mode: current response is empty, cannot score---------------------------------------------------\n",completion_msgs)
                    print("History:")
                    print(prompt_msgs)
                    print("Attack request:")
                    print(attacker_request)
                    print("---------------------------------------------------\n")
                    raise NotImplementedError(
                        "TRAIN_RH mode requires a non-empty current response")
                my_train_logger.debug("Entering GPTJudge branch")
                my_train_logger.debug("Input length: "+str(len(attacker_request)+len(current_res)))
                score, reason = judge.infer_single(
                    attacker_request, current_res)
                format_score = check_md_format_score(current_res)[0]
                if REWARD_WHITE:

                    payload = {
                        "instruction": attacker_request,
                        "completion": current_res
                    }

                    resp = requests.post("http://localhost:8800/evaluate", json=payload, timeout=300)
                    white_score = resp.json()["score"]
                    deferred_white[len(rewards)] = (
                        score, format_score, white_score)
                    rewards.append(None)
                    continue
                final_score = (5-score)/4
                final_score = (
                    format_score - 1) if format_score != 1.0 else final_score
                rewards.append(final_score)
                continue
            my_train_logger.debug("Entering non-fused branch")   
            if (isinstance(completion_msgs, str) and len(completion_msgs) < 20) or (current_res and len(current_res) < 20):
                print(
                    f"-------Mute penalty. Completion too short: \n {completion_msgs}\n---------------------------------------------------\n")
                final_score = -1.0
                deduction_reason = "too_short"
                rewards.append(final_score)

            elif ngram_repeat_check((current_res if current_res else completion_msgs)):
                print(
                    f"-------Repetition penalty. Completion has excessive repetition: \n{completion_msgs}\n---------------------------------------------------\n")
                final_score = -1.0
                deduction_reason = "repetition"
                rewards.append(final_score)

            else:
                safe_score = evaluate_safe_with_gptoss(
                    prompt_msgs+completion_msgs, attacker_request) if TRAIN_SAFE else evaluate_harm_with_gptoss(prompt_msgs+completion_msgs, attacker_request)

                # Format check
                format_score = check_md_format_score(current_res)[0]
                if TRAIN_SAFE and REWARD_WHITE:

                    payload = {
                        "instruction": attacker_request,
                        "completion": current_res
                    }

                    resp = requests.post("http://localhost:8800/evaluate", json=payload, timeout=300)
                    white_score = resp.json()["score"]
                    deferred_white[len(rewards)] = (
                        safe_score, format_score, white_score)
                    rewards.append(None)
                    continue
                final_score = (
                    format_score - 1) if format_score != 1.0 else safe_score
                if final_score == 0:
                    print(
                        f"-----Total score is 0, format_score: {format_score}, safe_score: {safe_score}----")
                rewards.append(final_score)


        # Normalize white scores per prompt group (z-score) and backfill: same granularity as GRPO group advantage,
        # so white only ranks generations within a prompt and never leaks across groups
        if deferred_white:
            # Consecutive identical prompts form a group (num_generations repeats)
            group_ids = []
            gid = -1
            for i in range(len(prompts)):
                if i == 0 or prompts[i] != prompts[i-1]:
                    gid += 1
                group_ids.append(gid)
            groups = {}
            for idx in deferred_white:
                groups.setdefault(group_ids[idx], []).append(idx)
            for idxs in groups.values():
                whites = [deferred_white[i][2] for i in idxs]
                w_mean = sum(whites) / len(whites)
                w_std = (
                    sum((w - w_mean) ** 2 for w in whites) / len(whites)) ** 0.5
                for i in idxs:
                    score_term, format_score, white_score = deferred_white[i]
                    white_norm = (white_score - w_mean) / (w_std + 1e-8)
                    final_score = ((2.5-score_term)/2.5+white_norm)/2
                    final_score = (
                        format_score - 1) if format_score != 1.0 else final_score
                    if final_score == 0:
                        print(
                            f"-----Total score is 0, format_score: {format_score}, safe_score: {score_term}----")
                    rewards[i] = final_score
            my_train_logger.debug(
                f"White per-group normalization: {len(groups)} groups, white counts per group={[len(v) for v in groups.values()]}")

    except Exception as e:
        print(f"----------------Error in reward_func, interrupted: {e}")
        raise e
    my_train_logger.debug(f"Batch rewards: {rewards}")
    return rewards


lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # init_lora_weights="gaussian",
    inference_mode=False,
    target_modules=["q_proj", "k_proj", "v_proj",
                    "o_proj", "gate_proj", "up_proj", "down_proj"],
)


def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

model = AutoModelForCausalLM.from_pretrained(model_name)

# print(model.config.eos_token_id)
# print(tokenizer.eos_token, tokenizer.eos_token_id)
# print(tokenizer.special_tokens_map)
# print(tokenizer.additional_special_tokens)
# exit()
# Check tokenizer/model consistency
if tokenizer.pad_token_id != model.config.pad_token_id:
    if model.config.pad_token_id is None:
        # For models without a pad token, set pad_token = eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        # tokenizer.pad_token = tokenizer.eos_token
        print(f"⚠️ Model has no PAD token. ")
        # print(f"⚠️ Model has no PAD token. Using EOS token ({model.config.eos_token_id}) as PAD token.")
    else:
        raise ValueError(
            f"Tokenizer PAD token ({tokenizer.pad_token_id}) "
            f"does not match model config PAD token ({model.config.pad_token_id}). "
            "Please use a tokenizer that matches the model."
        )
if tokenizer.eos_token_id != model.config.eos_token_id:
    raise ValueError(
        f"Tokenizer EOS token ({tokenizer.eos_token_id}) "
        f"does not match model config EOS token ({model.config.eos_token_id}). "
        "Please use a tokenizer that matches the model."
    )

if __name__ == "__main__":
    # DeepSpeed zero
    ds_cfg = {
        "zero_optimization": {
            "stage": 2,
            "overlap_comm": True,
            "reduce_bucket_size": "auto",
            "contiguous_gradients": True,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            },
        },
        "gradient_clipping": "auto",
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "steps_per_print": 1,

    }
    # https://github.com/huggingface/trl/pull/3617 has a bug; with vllm you may need to pass generation_kwargs
    
    training_args = GRPOConfig(
        # output_dir="test",
        output_dir=f"xl/BL-{model_name}_{'ronghe' if TRAIN_RH else 'safe' if TRAIN_SAFE else 'harm'}_V3_md_SafeMT_{per_device_train_batch_size}_{gradient_accumulation_steps}_{lr}{'_withRM' if REWARD_WHITE else ''}",
        # output_dir="llama_V2fangya_3ka_4_32_2e4_witheval",
        logging_steps=1, fp16=True,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        max_prompt_length=36864,
        max_completion_length=2048,
        save_total_limit=4,
        save_strategy="steps",
        save_steps=5,
        eval_strategy="steps",     # periodic evaluation
        eval_steps=5,
        per_device_eval_batch_size=per_device_eval_batch_size,
        load_best_model_at_end=True,
        report_to="wandb",
        do_eval=True,
        lr_scheduler_type="cosine",
        learning_rate=lr,
        deepspeed=ds_cfg,
        use_vllm=True,
        vllm_gpu_memory_utilization=0.45,
        vllm_mode="colocate",
        generation_kwargs={
        },
        num_generations=4,
        num_train_epochs=3,
    )

    trainer = GRPOTrainer(
        # model= "meta-llama/Llama-3.1-8B-Instruct",
        # model="Qwen/Qwen2.5-7B-Instruct",
        # model="/rds/projects/f/fengyv-proactive-ds/projects/SafeDialBench-Dataset/merged_model",/
        # model="Qwen/Qwen2.5-7B",
        model=model,
        # tokenizer=tokenizer,
        reward_funcs=reward_func,
        args=training_args,
        peft_config=lora_config,

    )
    try:
        # trainer.train(resume_from_checkpoint=True)
        trainer.train()
    except Exception as e:
        print(f"Training interrupted: {e}")
        raise e
