

# DEBUG = True
POST_PROCESS_FINAL_ANSWER = True
DEBUG = False
# POST_PROCESS_FINAL_ANSWER = False
import re
from fastapi import FastAPI, Request
import httpx
import uvicorn
import json
import sys

from proactive_defence import proactiveDefencer
from proactive_defence.utils.format_convert import OpenAIStyleResponse, messages_to_nested_list
from proactive_defence.utils.llms.gemini import GeminiChater
from proactive_defence.utils.keys_loader import load_api_keys_from_yaml
from IPython  import display
import os
# Gemini_KEYS = load_api_keys_from_yaml(os.path.expanduser("~/keys.yaml"))
# geminichater = GeminiChater(
#     api_keys=Gemini_KEYS,
#     model="gemini-2.5-flash-lite"  
# )
app = FastAPI()
defencer=proactiveDefencer(
        # safe_model_path="meta-llama/Meta-Llama-3.1-8B-Instruct",
        # harmful_model_path="meta-llama/Meta-Llama-3.1-8B-Instruct",
        vllm_api_base="http://localhost:8002/v1",
        vllm_api_key="EMPTY",
        use_vllm=True,
        # use_textgrad=True,
    )


VLLM_URL = "http://localhost:8000"  
VLLM_URL_llama = "http://localhost:8000" 
# VLLM_URL = "http://localhost:30010"  

def modify_payload(payload: dict) -> dict:


    payload['messages'][-1]['content']=defencer.analysis(messages_to_nested_list(payload['messages']),xy=0)
    if DEBUG:
        display.display(payload)
    return payload

from fastapi.responses import Response


# @app.get("/get_world_size/")
# def get_world_size():
#     return {"world_size": 1}




@app.get("/{path:path}")
async def proxy_get(path: str, request: Request):
    
    print("---------------------------------Received GET params---------------------------------")
    print(request.query_params)
    print("--------------------------------------------------------------\n")

  
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, read=600.0)) as client:
        vllm_response = await client.get(f"{VLLM_URL}/{path}", params=request.query_params)
    
    if DEBUG:

        print("---------------------------------VLLM response---------------------------------")
        print(vllm_response.text[:500])
        print("--------------------------------------------------------------\n")


    return Response(
        content=vllm_response.content,
        status_code=vllm_response.status_code,
        headers=vllm_response.headers
    )




@app.post("/{path:path}")
async def proxy(request: Request, path: str):
    try:
        data = await request.json()
    except Exception:
        data = await request.body()
    # print("---------------------------------Received data:---------------------------------\n", data,"--------------------------------------------------------------\n")
    # exit()

    if "STAIR"  not in data.get("model","") and isinstance(data, dict):
        data = modify_payload(data)
    if False:
    # if "emini" in data.get("model",""):
        pass
        # if data["messages"][0]["role"]!="system":
        #     raise NotImplementedError(" system + user ")
        # response = geminichater.generate_content(
        #     history=data["messages"][1:],
        #     contents=data["messages"][-1]['content'],
        #     system_instruction=data["messages"][0]['content'],
        # )
        # if DEBUG:
        #     display.display(response)
        # return OpenAIStyleResponse(response)
    else:

        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, read=600.0)) as client:
            vllm_response = await client.post(f'{ VLLM_URL_llama if "STAIR"  in data.get("model","") and "Llama"  in data.get("model","") else VLLM_URL }/{path}', json=data)
        
        if "STAIR" not in data.get("model",""):
            return Response(
            content=vllm_response.content,
            status_code=vllm_response.status_code,
            headers=vllm_response.headers
            )



        
        if DEBUG:

            display.display("status:", vllm_response.status_code)
            display.display("headers:", vllm_response.headers)
            display.display("text:", vllm_response.text)
            print("--------------------------------------\n")
        response_json = vllm_response.json()
        if POST_PROCESS_FINAL_ANSWER:

            
            content = response_json["choices"][0]["message"]["content"]

            #  Final Answer
            match = re.search(r"Final Answer:\s*(.*)", content, re.DOTALL)
            final_answer = match.group(1).strip() if match else content  
            response_json["choices"][0]["message"]["content"] = final_answer





        # return vllm_response.json()
        return response_json


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
