import sys
import json
import traceback
from openai import OpenAI
from proactive_defence.prompts.defence_prompts_md import *
import transformers
import torch
import re
from proactive_defence.prompts.tg_prompts import HARMFUL_EVALUATION_SYSTEM_INSTRUCTION, SAFE_EVALUATION_SYSTEM_INSTRUCTION
# from .prompts.defence_prompts import *
import logging
import textgrad as tg
from .utils.llms.tg_openai import TgMyChatOpenAI

from tgd import BlackboxLLMWithHistory
logger1 = logging.getLogger("log1")
logger2 = logging.getLogger("log2")
FAST=True
if not logger1.hasHandlers():
    handler1 = logging.FileHandler("logsafe.txt", encoding="utf-8")
    formatter = logging.Formatter("%(message)s")
    handler1.setFormatter(formatter)
    logger1.addHandler(handler1)
    logger1.setLevel(logging.INFO)

if not logger2.hasHandlers():
    handler2 = logging.FileHandler("logharm.txt", encoding="utf-8")
    formatter = logging.Formatter("%(message)s")
    handler2.setFormatter(formatter)
    logger2.addHandler(handler2)
    logger2.setLevel(logging.INFO)


def do_something1(content):
    logger1.info(content)
    for handler in logger1.handlers:
        handler.flush()


def do_something2(content):
    logger2.info(content)
    for handler in logger2.handlers:
        handler.flush()


class proactiveDefencer:
    def __init__(
        self,
        safe_model_path="meta-llama/Meta-Llama-3.1-8B-Instruct",
        harmful_model_path="meta-llama/Meta-Llama-3.1-8B-Instruct",
        vllm_api_base="http://localhost:8000/v1",
        vllm_api_key="EMPTY",
        use_vllm=False,
        use_textgrad=False,
    ):
        self.safe_model_path = safe_model_path
        self.harmful_model_path = harmful_model_path
        self._safe_generator = None
        self._harmful_generator = None
        self._safe_generator_local = None
        self._harmful_generator_local = None
        self.use_vllm = use_vllm
        if use_vllm:
            self.client = OpenAI(base_url=vllm_api_base, api_key=vllm_api_key)
        self.use_textgrad = use_textgrad
        if use_textgrad:
            Tg_model_name = 'Qwen/Qwen2.5-32B-Instruct'
            Tg_model_url = 'http://localhost:30009/v1'

            engine = TgMyChatOpenAI(
                model_string=Tg_model_name, base_url=Tg_model_url)
            tg.set_backward_engine(engine, override=True)
            # self.safe_client= tg.BlackboxLLM(TgMyChatOpenAI('deepseek-ai/deepseek-llm-67b-chat', base_url='http://localhost:8000/v1',system_prompt=SAFE_USER_REQUEST_PREDICTION)) # response model
            self.safe_client = BlackboxLLMWithHistory(TgMyChatOpenAI(
                Tg_model_name, base_url=Tg_model_url, system_prompt=SAFE_USER_REQUEST_PREDICTION))  # response model
            self.harmful_client = BlackboxLLMWithHistory(TgMyChatOpenAI(
                Tg_model_name, base_url=Tg_model_url, system_prompt=POTENTIALLY_HARMFUL_GENERATION))  # response model

    def _load_model(self, model_path, generator_attr):
        # if self.use_vllm:
        #     return
        if getattr(self, generator_attr) is None:
            print(f"Loading model from {model_path} ...")
            pipeline = transformers.pipeline(
                "text-generation",
                model=model_path,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
            )
            setattr(self, generator_attr, pipeline)

    def _generate(self, messages, max_new_tokens=2048, generator_attr="_safe_generator", model_path=None):
        if self.use_vllm:
            try:
                response = self.client.chat.completions.create(
                    model=generator_attr,
                    messages=messages,
                )
            except Exception as e:
                print(f"Defence vllm request failed, messages:", messages)
                traceback.print_exc()
                raise e
            output = response.choices[0].message.content
            # print(
            if ((output.strip() == '') or len(output) < 20) and (generator_attr == "_harmful_generator"):
                do_something2(json.dumps(messages))
                if FAST:
                    return output
            elif generator_attr == "_safe_generator" and response.usage.completion_tokens > 1000:
                do_something1(json.dumps(messages))
                do_something1(output)
                output = output[:10000]
                if FAST:
                    return output
            else:
                return output

        self._load_model(model_path, generator_attr+"_local")
        generator = getattr(self, generator_attr+"_local")
        outputs = generator(messages, max_new_tokens=max_new_tokens)
        print(
            f'--------------------------------------------------------\nvllm response error, local response:\n {outputs[0]["generated_text"][-1]["content"]}\n--------------------------------------------------------')
        return outputs[0]["generated_text"][-1]["content"]

    def _build_message_sequence(self, messages, system_prompt):
        temp_messages = [{"role": "system", "content": system_prompt}]
        for message in messages[:-1]:
            temp_messages.append({
                "role": message[0],
                "content": message[1]
            })
        return temp_messages
    
    def safe_user_request_prediction(self, messages):
        history = self._build_message_sequence(
            messages,
            SAFE_USER_REQUEST_PREDICTION
        )
        current_request = [{
            "role": "user",
                    "content": f"""<Instructions>\n{PROMPT_SAFE["Instructions"]}\n</Instructions>"""
        }]
        if not self.use_textgrad:
            return self._generate(
                history+current_request,
                generator_attr="_safe_generator",
                model_path=self.safe_model_path
            )
        question = tg.Variable(current_request[0]['content'],
                               role_description="question to the LLM",
                               requires_grad=True)

        optimizer = tg.TGD(parameters=[question])
        evaluation_instruction = tg.Variable(SAFE_EVALUATION_SYSTEM_INSTRUCTION.substitute(history=str(history[1:])),
                                             requires_grad=False,
                                             role_description="system prompt")

        loss_fn = tg.TextLoss(evaluation_instruction)
        # Step 3: Do the loss computation, backward pass, and update the punchline.
# Exact same syntax as PyTorch!
        for _ in range(3):
            answer = self.safe_client(question, history=history[1:])
            optimizer.zero_grad()
            loss = loss_fn(answer)
            if re.search(r"<ACCURACY>\s*1\s*</ACCURACY>", loss.value) or _ == 2:
                break
            loss.backward()
            optimizer.step()
        return answer.value

    def potentially_harmful_generation(self, messages):

        history = self._build_message_sequence(
            messages,
            POTENTIALLY_HARMFUL_GENERATION
        )

        if not self.use_textgrad:
            return self._generate(history+[{
                "role": "user",
                "content": f"""<User_Input>\n{messages[-1][1]}\n</User_Input>\n\n<Instructions>\n{PROMPT_HARM["Instructions"]}\n</Instructions>"""
            }], generator_attr="_harmful_generator",
                model_path=self.harmful_model_path)
        question = tg.Variable(PROMPT_HARM["Instructions"],
                               role_description="current request to the LLM",
                               requires_grad=True)

        optimizer = tg.TGD(parameters=[question])
        
        evaluation_instruction = tg.Variable(HARMFUL_EVALUATION_SYSTEM_INSTRUCTION.substitute(history=str(history[1:]+[{
                "role": "user",
                "content": messages[-1][1]
            }])),
                                             requires_grad=False,
                                             role_description="system prompt")

        loss_fn = tg.TextLoss(evaluation_instruction)
        current_request = str([{
                "role": "user",
                "content": f"""<User_Input>\n{messages[-1][1]}\n</User_Input>\n\n<Instructions>\n{PROMPT_HARM["Instructions"]}\n</Instructions>"""
            }])

        marker = str(PROMPT_HARM["Instructions"])

        # partition yields three parts: prefix, separator, suffix
        prefix, _, suffix = current_request.partition(marker)
        current_request_prefix= tg.Variable(prefix,
                               role_description="prefix to the current request",
                               requires_grad=False)
        current_request_suffix= tg.Variable(suffix,
                               role_description="suffix to the current request",
                               requires_grad=False)

        # Step 3: Do the loss computation, backward pass, and update the punchline.
# Exact same syntax as PyTorch!
        for _ in range(3):
            current_request = current_request_prefix + question + current_request_suffix
            answer = self.harmful_client(current_request, history=history[1:])
            optimizer.zero_grad()
            loss = loss_fn(answer)
            if re.search(r"<ACCURACY>\s*1\s*</ACCURACY>", loss.value) or _ == 2:
                break
            loss.backward()
            optimizer.step()
        return answer.value

    def analysis(self, messages, xy=0):
        if messages[0][0] == 'system':
            messages = messages[1:]
        if xy == 1:
            if len(messages) < 3:
                return messages[-1][1]
            return XR1_AGGREGATE_ANALYSIS_TEMPLATE.substitute(
                user_input=messages[-1][1],
                safe_result=self.safe_user_request_prediction(messages)
            )
        elif xy == 2:
            return XR2_AGGREGATE_ANALYSIS_TEMPLATE.substitute(
                user_input=messages[-1][1],
                harmful_result=self.potentially_harmful_generation(messages),
            )
        elif xy == 0:
            if len(messages) < 3:
                return XR2_AGGREGATE_ANALYSIS_TEMPLATE.substitute(
                    user_input=messages[-1][1],
                    harmful_result=self.potentially_harmful_generation(
                        messages),
                )
            return AGGREGATE_ANALYSIS_TEMPLATE.substitute(
                user_input=messages[-1][1],
                safe_result=self.safe_user_request_prediction(messages),
                harmful_result=self.potentially_harmful_generation(messages),
            )
    def analysis_MD(self, messages, xy=0):
        if xy!=0:
            raise NotImplementedError("Only xy=0 is implemented for MD analysis.")
        if messages[0][0] == 'system':
            messages = messages[1:]
        
        analysis = AGGREGATE_ANALYSIS_TEMPLATE_MD.substitute(
                user_input=messages[-1][1],
                safe_result=self.safe_user_request_prediction(messages),
                harmful_result=self.potentially_harmful_generation(messages),
            )
        return analysis
