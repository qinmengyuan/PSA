
import os
import platformdirs
from openai import OpenAI
import textgrad as tg
# Import the base class
import sys
sys.path.append('/data/psa/x-teaming')
from tgd import ChatOpenAIWithHistory, BlackboxLLMWithHistory
from textgrad.engine.openai import ChatOpenAI

class TgMyChatOpenAI(ChatOpenAIWithHistory):
    """
    Modified version that inherits from ChatOpenAIWithHistory with updated base_url handling
    """
    
    def __init__(
        self,
        model_string: str = "gpt-3.5-turbo-0613",
        system_prompt: str = ChatOpenAI.DEFAULT_SYSTEM_PROMPT,
        is_multimodal: bool = False,
        base_url: str = None,
        **kwargs
    ):
        """
        :param model_string:
        :param system_prompt:
        :param base_url: Used to support Ollama or other OpenAI-compatible APIs
        """
        # Initialize history messages
        self.history_messages = []
        
        # Cache setup
        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_openai_{model_string}.db")
        
        # Initialize cache from parent class
        super(ChatOpenAI, self).__init__(cache_path=cache_path)
        
        self.system_prompt = system_prompt
        self.base_url = base_url
        
        # Modified base_url handling logic
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
    def _check_cache(self, prompt: str):
            return None

if __name__ == "__main__":
    # Test code
    tg_openai = TgMyChatOpenAI(model_string='deepseek-ai/deepseek-llm-67b-chat', base_url='http://localhost:8000/v1')
    print(tg_openai.generate("Hello, how are you?"))
    tg.set_backward_engine(tg_openai, override=True)
    model= BlackboxLLMWithHistory(TgMyChatOpenAI('deepseek-ai/deepseek-llm-67b-chat', base_url='http://localhost:8000/v1',system_prompt='you are an assistant'))
    question_string = ("If it takes 1 hour to dry 25 shirts under the sun, "
                   "how long will it take to dry 30 shirts under the sun? "
                   "Reason step by step")

    question = tg.Variable(question_string,
                        role_description="question to the LLM",
                        requires_grad=False)

    answer = model(question)
