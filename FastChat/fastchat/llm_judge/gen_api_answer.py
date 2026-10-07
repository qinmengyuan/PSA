"""Generate answers with language model APIs

Usage:
python3 gen_api_answer.py --model gpt-3.5-turbo --language en
python3 gen_api_answer.py --model gpt-3.5-turbo --language zh

python gen_api_answer.py --model gpt-oss120  --language en --parallel 5
python gen_api_answer.py --model meta-llama/Meta-Llama-3-8B-Instruct  --language en --parallel 20
"""
import argparse
import copy
import json
import os
import time
import concurrent.futures
import random
import openai
import shortuuid
import tqdm

from fastchat.llm_judge.common import (
    chat_completion_myopenai,
    load_questions,
    temperature_config,
    chat_completion_openai,
    chat_completion_anthropic,
    chat_completion_palm,
    chat_completion_moonshot,
    chat_completion_deepseek,
    chat_completion_qwen,
)
from fastchat.model.model_adapter import get_conversation_template, ANTHROPIC_MODEL_LIST, MOONSHOT_MODEL_LIST, DEEPSEEK_MODEL_LIST, QWEN_MODEL_LIST

from proactive_defence import proactiveDefencer


DEFENCE_ENABLED = False  # Set to True to enable defence against unsafe content

if DEFENCE_ENABLED:
    # defencer = proactiveDefencer(
    #     model_name="meta-llama/Meta-Llama-3.1-8B-Instruct")
    defencer = proactiveDefencer(
        # safe_model_path="/rds/projects/f/fengyv-proactive-ds/projects/SafeDialBench-Dataset/Qwen2.5-7B-Instruct_safe_biglr.lr2e5/checkpoint-180",
        # harmful_model_path="/rds/projects/f/fengyv-proactive-ds/projects/SafeDialBench-Dataset/IT_V2fangya_3ka_4_16_2e5_witheval/checkpoint-330/",
        vllm_api_base="http://localhost:8002/v1",
        use_vllm=True,
        # use_textgrad=True,
    )

def get_finished_qids(answer_file):
    finished = set()
    if os.path.exists(answer_file):
        with open(answer_file, "r", encoding="utf-8") as fin:
            for l in fin:
                try:
                    ans = json.loads(l)
                    finished.add(ans["id"])
                except Exception:
                    continue
    return finished


def get_answer(
    question: dict, model: str, num_choices: int, max_tokens: int, answer_file: str
):
    assert (
        args.force_temperature is not None and "required_temperature" in question.keys()
    ) == False
    if args.force_temperature is not None:
        temperature = args.force_temperature
    # elif "required_temperature" in question.keys():
    #     temperature = question["required_temperature"]
    # elif question["category"] in temperature_config:
    #     temperature = temperature_config[question["category"]]
    else:
        temperature = 0.7

    choices = []
    chat_state = None  # for palm-2 model
    for i in range(num_choices):
        conv = get_conversation_template(model)

        turns = []
        history = question["history"]

        for idx, turn in enumerate(history):
            user_msg = turn["user"]
            bot_msg = turn.get("bot", None)

            conv.append_message(conv.roles[0], user_msg)
            if DEFENCE_ENABLED:
                    user_msg = defencer.analysis_MD(conv.messages,xy=0)
                    

                    if random.randint(1, 1000) == 1:

                        print(f"User message after defence: {user_msg}")
                    conv_copy = copy.deepcopy(conv)
                    conv.messages.pop()  # remove the last message
                    conv.append_message(conv.roles[0], user_msg)
            conv.append_message(conv.roles[1], None)
            # prompt = conv.get_prompt()
            # print(idx,": ", prompt)

            if model in ANTHROPIC_MODEL_LIST:
                output = chat_completion_anthropic(model, conv, temperature, max_tokens)
            elif model == "palm-2-chat-bison-001":
                chat_state, output = chat_completion_palm(
                    chat_state, model, conv, temperature, max_tokens
                )
            elif model in MOONSHOT_MODEL_LIST:
                output = chat_completion_moonshot(model, conv, temperature, max_tokens)
            elif model in DEEPSEEK_MODEL_LIST:
                output = chat_completion_deepseek(model, conv, temperature, max_tokens)
            elif model in QWEN_MODEL_LIST:
                output = chat_completion_qwen(model, conv, temperature, max_tokens)
            elif "STAIR" in model:
                output = chat_completion_myopenai(model, conv, api_dict={"api_base": "http://localhost:5000/v1", "api_key": "EMPTY"})
            else:
                output = chat_completion_myopenai(model, conv, api_dict={"api_base": "http://localhost:8000/v1", "api_key": "EMPTY"})
            if DEFENCE_ENABLED:
                conv = copy.deepcopy(conv_copy)
                conv.append_message(conv.roles[1], None)
            conv.update_last_message(output)

            turns.append(output)
            # print(output)
            # print("****", turns)
        choices.append({"index": i, "turns": turns})

    # Dump answers
    ans = {
        "id": question["id"],
        "task": question["task"],
        "answer_id": shortuuid.uuid(),
        "model_id": model,
        "method": question["method"],
        "choices": choices,
        "tstamp": time.time(),
    }

    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    with open(answer_file, "a", encoding="utf-8") as fout:
        fout.write(json.dumps(ans, ensure_ascii=False) + "\n")


def reorg_answer_file(answer_file):
    """Sort and deduplicate answers by question ID"""
    answers = {}
    with open(answer_file, "r", encoding="utf-8") as fin:
        for l in fin:
            ans = json.loads(l)
            qid = ans["id"]
            answers[qid] = l

    qids = sorted(answers.keys())
    with open(answer_file, "w", encoding="utf-8") as fout:
        for qid in qids:
            fout.write(answers[qid])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench-name",
        type=str,
        default="SafeDial_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--force-temperature", type=float, help="Forcibly set a sampling temperature."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument(
        "--parallel", type=int, default=1, help="The number of concurrent API calls."
    )
    parser.add_argument("--openai-api-base", type=str, default=None)
    parser.add_argument(
        "--language", 
        type=str, 
        default="zh", 
        choices=["zh", "en"],
        help="Language of the dataset (zh for Chinese, en for English)"
    )
    args = parser.parse_args()

    # if args.openai_api_base is not None:
    #     openai.api_base = args.openai_api_base

    # Set dataset file based on language
    if args.language == "zh":
        question_file = f"data/{args.bench_name}/datasets_zh.jsonl"
    else:
        question_file = f"data/{args.bench_name}/questions_en.jsonl"
    
    questions = load_questions(question_file, args.question_begin, args.question_end)

    # Set output file based on language
    if args.answer_file:
        answer_file = args.answer_file
    else:
        if args.language == "zh":
            answer_file = f"data/{args.bench_name}/model_answer/{args.model}_zh.jsonl"
        else:
            answer_file = f"data/{args.bench_name}/model_answer/{args.model}_en.jsonl"
    
    print(f"Output to {answer_file}")
    
    # Added: filter out finished questions
    finished_qids = get_finished_qids(answer_file)
    questions = [q for q in questions if q["id"] not in finished_qids]
    if not questions:
        print("All questions are finished; nothing to do.")
        exit()

    random.shuffle(questions)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = []
        for question in questions:
            future = executor.submit(
                get_answer,
                question,
                args.model,
                args.num_choices,
                args.max_tokens,
                answer_file,
            )
            futures.append(future)

        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()

    reorg_answer_file(answer_file)