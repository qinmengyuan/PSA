
import sys

sys.path.append('/data/psa/khgfds/')
from tqdm import tqdm  # progress bar library
from concurrent.futures import ProcessPoolExecutor, as_completed
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
from typing import List, Dict
import re
from json_repair import repair_json
import json
from proactive_defence.utils.SafetyScorer import SafetyScorer
from proactive_defence.prompts.defence_prompts_md import AGGREGATE_ANALYSIS_TEMPLATE_SYS_MD, POTENTIALLY_HARMFUL_GENERATION, PROMPT_HARM, PROMPT_SAFE, SAFE_USER_REQUEST_PREDICTION
from FastChat.fastchat.model.model_adapter import get_conversation_template
from FastChat.fastchat.llm_judge.common import load_questions
from filelock import FileLock
from transformers import AutoTokenizer
from trl import apply_chat_template
from tqdm import tqdm
import torch
from proactive_defence.utils.format_convert import messages_to_nested_list
from proactive_defence import proactiveDefencer
from datasets import load_dataset, Dataset
from collections import Counter
from datetime import datetime
import sqlite3
import os
import yaml
import pickle
from proactive_defence.utils import load_config

config = load_config()

TRAIN_SAFE = config["TRAIN_SAFE"]
TRAIN_RH = config["TRAIN_RH"]
model_name = config["model_name"]
per_device_train_batch_size = config["per_device_train_batch_size"]
gradient_accumulation_steps = config["gradient_accumulation_steps"]
lr = config["learning_rate"]
save_dir = config["cache_pickle_write_dir"]
print(TRAIN_SAFE, TRAIN_RH)


defencer = proactiveDefencer(
    # safe_model_path="meta-llama/Meta-Llama-3.1-8B-Instruct",
    # harmful_model_path="meta-llama/Meta-Llama-3.1-8B-Instruct",
    vllm_api_base="http://localhost:8002/v1",
    vllm_api_key="EMPTY",
    use_vllm=True,
    # use_textgrad=True,
)

# Always load the tokenizer from the official / model repo
tokenizer = AutoTokenizer.from_pretrained(model_name)


def prepare_datasaets(dataset_name, data_file=r'/data/psa/khgfds/FastChat/fastchat/llm_judge/data/SafeDial_bench/questions_en.jsonl', questions=None):
    if dataset_name == "SafeDial_bench":
        global tokenizer
        # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

        if questions is None:
            questions = load_questions(
                data_file)
        formatted_data = []

        for question in tqdm(questions):
            choices = []
            for i in range(1):
                torch.manual_seed(i)
                conv = get_conversation_template("gpt-4")
                turns = []
                history = question["history"]
                method = question["method"]

                for idx, turn in enumerate(history):

                    user_msg = turn["user"]
                    bot_msg = turn.get("bot", None)
                    if (False if TRAIN_RH else idx < 3 if TRAIN_SAFE else idx < 1):
                        conv.append_message(conv.roles[0], user_msg)
                        conv.append_message(conv.roles[1], bot_msg)
                        continue

                    # formatted_data.append(apply_chat_template(
                    #     {"prompt": ([{"role": "system", "content": SAFE_USER_REQUEST_PREDICTION if TRAIN_SAFE else POTENTIALLY_HARMFUL_GENERATION}]+conv.to_deepseek_api_messages()+[{
                    #         "role": "user",
                    #         "content": ('' if TRAIN_SAFE else ('<User_Input>\n'+user_msg+'\n</User_Input>\n'))+f"""<Instructions>\n{PROMPT_SAFE["Instructions"] if TRAIN_SAFE else PROMPT_HARM["Instructions"]}\n</Instructions>"""
                    #     }])}, tokenizer))
                    data = {"prompt": ([{"role": "system", "content": AGGREGATE_ANALYSIS_TEMPLATE_SYS_MD if TRAIN_RH else SAFE_USER_REQUEST_PREDICTION if TRAIN_SAFE else POTENTIALLY_HARMFUL_GENERATION}]+conv.to_deepseek_api_messages()),
                            "mytask": question["task"],
                            **({"attacker_requests": user_msg} if True or (TRAIN_SAFE or TRAIN_RH) else {})}
                    # print(question["task"])
                    # print(type(question["task"]))
                    # exit()
                    if TRAIN_RH:
                        conv.append_message(conv.roles[0], user_msg)
                        data["prompt"] = data["prompt"]+([{"role": "user",
                                                          "content": messages_to_nested_list(conv.to_deepseek_api_messages())}])
                    else:
                        data["prompt"] = data["prompt"]+[{
                            "role": "user",
                            "content": ('' if TRAIN_SAFE else ('<User_Input>\n'+user_msg+'\n</User_Input>\n'))+f"""<Instructions>\n{PROMPT_SAFE["Instructions"] if TRAIN_SAFE else PROMPT_HARM["Instructions"]}\n</Instructions>"""
                        }]
                    formatted_data.append(data)
                    # print(formatted_data[0])
                    # exit()
                    if not TRAIN_RH:
                        conv.append_message(conv.roles[0], user_msg)
                    conv.append_message(conv.roles[1], bot_msg)
        return formatted_data


def split_dataset(dataset, train_ratio=0.8):
    """
    Split the dataset into train and eval sets.

    Args:
        dataset: the full dataset
        train_ratio: fraction used for training, default 0.8

    Returns:
        train_dataset, eval_dataset
    """
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)

    # Shuffle the dataset randomly
    import random
    random.seed(42)  # fix the seed for reproducibility
    shuffled_indices = list(range(total_size))
    random.shuffle(shuffled_indices)

    train_indices = shuffled_indices[:train_size]
    eval_indices = shuffled_indices[train_size:]

    train_dataset = [dataset[i] for i in train_indices]
    eval_dataset = [dataset[i] for i in eval_indices]

    return train_dataset, eval_dataset


def process_item(item, key='prompt'):
    if not TRAIN_RH:
        return item
    item[key][-1]["content"] = defencer.analysis_MD(
        item[key][-1]["content"], xy=0)
    return item


def parallel_process(dataset, max_workers=8, key='prompt'):
    processed_dataset = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_item, item, key=key) for item in dataset]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing", unit="item"):
            processed_dataset.append(future.result())
    return processed_dataset

if __name__ == "__main__":
    # ------------------------New split
    # Paper setting: randomly sample 15% of the full SafeDialBench (2,037 items) for training-data construction,
    # the remaining 85% is reserved for evaluation only. Fixed seed for reproducibility.
    import random
    random.seed(42)
    all_questions = load_questions(
        "/data/psa/khgfds/FastChat/fastchat/llm_judge/data/SafeDial_bench/questions_en.jsonl")
    random.shuffle(all_questions)
    n_train = int(len(all_questions) * 0.15)
    train_questions = all_questions[:n_train]
    eval_pool = all_questions[n_train:]
    print(f"Total {len(all_questions)} items; sampled 15%={n_train} for training data, 85%={len(eval_pool)} reserved for evaluation")

    train_dataset = prepare_datasaets(
        "SafeDial_bench", questions=train_questions)
    eval_dataset = prepare_datasaets(
        "SafeDial_bench", questions=eval_pool[:10])


    # for i in train_dataset:
    #     i['prompt'][-1]["content"] = defencer.analysis_MD(
    #         i['prompt'][-1]["content"], xy=0)
    # for i in eval_dataset:
    #     i['prompt'][-1]["content"] = defencer.analysis_MD(
    #         i['prompt'][-1]["content"], xy=0)


    train_dataset = parallel_process(train_dataset, max_workers=20)
    eval_dataset = parallel_process(eval_dataset, max_workers=20)

    os.makedirs(save_dir, exist_ok=True)
    train_path = os.path.join(save_dir, "train_dataset.pkl")
    eval_path = os.path.join(save_dir, "eval_dataset.pkl")

    # Save
    with open(train_path, "wb") as f:
        pickle.dump(train_dataset, f)

    with open(eval_path, "wb") as f:
        pickle.dump(eval_dataset, f)

    print("Saved:")
    print(train_path)
    print(eval_path)
