# ASCEND_RT_VISIBLE_DEVICES=1  uvicorn  rewarder_fw:app --host 0.0.0.0   --port 8800   --workers 1


import asyncio
import torch
import torch_npu
import torch.nn as nn
from collections import deque
from typing import Optional, List
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import (
    PreTrainedModel,
    LlamaConfig,
    LlamaModel,
    LlamaTokenizer,
)

# ======================
# Reward Model
# ======================
class LlamaRewardModel(PreTrainedModel):
    config_class = LlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.regression_head = nn.Linear(
            self.config.hidden_size, 1, bias=False
        )


    def forward(  # args are the same as LlamaForCausalLM
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        transformer_outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=False
            )
        hidden_states = transformer_outputs[0]
        rewards = self.regression_head(hidden_states).squeeze(-1)

        ends = attention_mask.cumsum(dim=1).argmax(
            dim=1).view(-1, 1).to(rewards.device)
        rewards = torch.gather(rewards, 1, ends)

        return rewards


# ======================
# Model Load
# ======================
ultrarm_template = """Human: {instruction}

Assistant: {completion}"""

ULTRARM_PATH = "/data/huancun/models/ms_cache/openbmb/UltraRM-13b"
tokenizer = LlamaTokenizer.from_pretrained(ULTRARM_PATH)
tokenizer.pad_token = tokenizer.eos_token
model = LlamaRewardModel.from_pretrained(
    ULTRARM_PATH,
    torch_dtype=torch.float16,
    device_map="npu:0"
)
model.eval()

# ======================
# FastAPI
# ======================
app = FastAPI(title="UltraRM Batch Reward Server")

class Request(BaseModel):
    instruction: str
    completion: str

# Request queue
queue = deque()
MAX_BATCH = 32
BATCH_TIMEOUT = 0.02  # 20ms


@app.post("/evaluate")
async def evaluate(req: Request):
    loop = asyncio.get_event_loop()
    future = loop.create_future()
    queue.append((req, future))
    return await future


# ======================
# Batch Worker
# ======================
async def batch_worker():
    while True:
        await asyncio.sleep(BATCH_TIMEOUT)

        if not queue:
            continue

        batch, futures = [], []
        while queue and len(batch) < MAX_BATCH:
            req, fut = queue.popleft()
            batch.append(req)
            futures.append(fut)

        prompts = [
            ultrarm_template.format(
                instruction=r.instruction,
                completion=r.completion
            )
            for r in batch
        ]

        inputs = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        ).to(model.device)

        with torch.inference_mode():
            scores = model(**inputs).squeeze(1).tolist()

        for fut, score in zip(futures, scores):
            fut.set_result({"score": score})


@app.on_event("startup")
async def startup():
    asyncio.create_task(batch_worker())
