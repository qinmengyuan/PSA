import transformers
import torch
from abc import ABC, abstractmethod


class LazyHuggingFaceModel(ABC):
    def __init__(self, model_name):
        self.model_name = model_name
        self._loaded = False

    def _load(self):
        if not self._loaded:
            self.load_model()
            self._loaded = True

    @abstractmethod
    def load_model(self):
        """
        Implemented by subclasses: load the model and related resources.
        """
        pass

    @abstractmethod
    def _generate(self, *args, **kwargs):
        """
        Implemented by subclasses: model inference call.
        """
        pass

    def generate(self, *args, **kwargs):
        self._load()
        return self._generate(*args, **kwargs)


class LazyMetaLlama(LazyHuggingFaceModel):
    def load_model(self):
        print(f"Loading pipeline for {self.model_name} ...")
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

    def _generate(self, prompt, max_new_tokens=2048):
        outputs = self.pipeline(prompt, max_new_tokens=max_new_tokens)
        return outputs[0]["generated_text"][-1]


class LazySomeOtherModel(LazyHuggingFaceModel):
    def load_model(self):
        print(f"Loading tokenizer and model for {self.model_name} ...")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_name)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

    def _generate(self, prompt, max_new_tokens=2048):
        inputs = self.tokenizer(
            prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)



class LazyDeepSeekChat(LazyHuggingFaceModel):
    def load_model(self):
        print(f"Loading DeepSeek Chat model from {self.model_name} ...")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_name)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()

        # Load the generation config and set pad_token_id
        self.model.generation_config = transformers.GenerationConfig.from_pretrained(
            self.model_name)
        self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id

    def _generate(self, messages, max_new_tokens=2048, **kwargs):
        # Apply the chat template and build the input
        input_tensor = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        # Run inference
        outputs = self.model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens
        )

        # Decode only the newly generated tokens
        generated_tokens = outputs[0][input_tensor.shape[1]:]
        result = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True)
        return result


if __name__ == "__main__":
    # Test examples
    llama = LazyMetaLlama("meta-llama/Meta-Llama-3.1-8B-Instruct")
    print(llama.generate("Hello, world!"))

    other = LazySomeOtherModel("gpt2")
    print(other.generate("Hello, world!"))
