from IPython.display import display
import textgrad as tg
from textgrad.engine.openai import ChatOpenAI, OLLAMA_BASE_URL
import os, platformdirs
try:
    from openai import OpenAI, AzureOpenAI
except ImportError:
    raise ImportError("If you'd like to use OpenAI models, please install the openai package by running `pip install openai`, and add 'OPENAI_API_KEY' to your environment variables.")

class TgMyChatOpenAI(ChatOpenAI):
    def __init__(
        self,
        model_string: str="gpt-3.5-turbo-0613",
        system_prompt: str=ChatOpenAI.DEFAULT_SYSTEM_PROMPT,
        is_multimodal: bool=False,
        base_url: str=None,
        **kwargs
    ):
        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_openai_{model_string}.db")

        super(ChatOpenAI, self).__init__(cache_path=cache_path)  # note: skip the parent __init__

        self.system_prompt = system_prompt
        self.base_url = base_url

        if not base_url:
            self.client = OpenAI(  # use the originally referenced OpenAI directly
                api_key=os.getenv("OPENAI_API_KEY")
            )
        elif base_url:
            self.client = OpenAI(
                base_url=base_url,
                api_key="ollama"
            )
        else:
            # ignore instead of raising
            pass

        self.model_string = model_string
        self.is_multimodal = is_multimodal

if __name__ == "__main__":
    # Test code
    tg_openai = TgMyChatOpenAI(model_string='deepseek-ai/deepseek-llm-67b-chat', base_url='http://localhost:8000/v1')
    print(tg_openai.generate("Hello, how are you?"))
    tg.set_backward_engine(tg_openai, override=True)
    model= tg.BlackboxLLM(TgMyChatOpenAI('deepseek-ai/deepseek-llm-67b-chat', base_url='http://localhost:8000/v1',system_prompt='you are an assistant'))
    question_string = ("If it takes 1 hour to dry 25 shirts under the sun, "
                   "how long will it take to dry 30 shirts under the sun? "
                   "Reason step by step")

    question = tg.Variable(question_string,
                        role_description="question to the LLM",
                        requires_grad=False)

    answer = model(question)

    answer.set_role_description("concise and accurate answer to the question")

    # Step 2: Define the loss function and the optimizer, just like in PyTorch!
    # Here, we don't have SGD, but we have TGD (Textual Gradient Descent)
    # that works with "textual gradients".
    optimizer = tg.TGD(parameters=[answer])
    evaluation_instruction = (f"Here's a question: {question_string}. "
                            "Evaluate any given answer to this question, "
                            "be smart, logical, and very critical. "
                            "Just provide concise feedback.")
    

    # TextLoss is a natural-language specified loss function that describes
    # how we want to evaluate the reasoning.
    loss_fn = tg.TextLoss(evaluation_instruction)
    # Step 3: Do the loss computation, backward pass, and update the punchline.
    # Exact same syntax as PyTorch!
    loss = loss_fn(answer)
    loss.backward()
    optimizer.step()
    display(answer)

    for _ in range(3):
        optimizer.zero_grad()
        loss = loss_fn(answer)
        loss.backward()
        optimizer.step()