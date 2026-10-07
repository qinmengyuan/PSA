from typing import List, Optional, Dict


def messages_to_nested_list(messages):
    return [[msg["role"], msg["content"]] for msg in messages]


def normalize_content_input(data: Dict) -> Dict:
    """
    Normalize a message dict into Gemini SDK-compatible format.

    Rules:
    - Ensures 'role' is either 'user' or 'model'. 'assistant' -> 'model'.
    - Converts 'content' or 'text' into 'parts' if not already present.
    - Keeps existing 'parts' if provided.
    """
    if not isinstance(data, dict):
        raise TypeError("Input must be a dictionary")

    #  role
    role = data.get("role", "user")
    if role not in ("user", "model"):
        if role == "assistant":
            role = "model"
        else:
            raise ValueError(
                f"Invalid role '{role}'. Must be 'user' or 'model'.")

    #  parts
    if "parts" in data and isinstance(data["parts"], list):
        parts = data["parts"]
    elif "content" in data:
        parts = [{"text": data["content"]}]
    elif "text" in data:
        parts = [{"text": data["text"]}]
    else:
        raise ValueError("Input must contain 'parts', 'content', or 'text'.")

    return {
        "role": role,
        "parts": parts
    }

#  wrapper 
class OpenAIStyleResponse:
    def __init__(self, original_response):
        self.original_response = original_response
        #  .choices 
        self.choices = [self.CandidateWrapper(c) for c in original_response['response'].candidates]

    class CandidateWrapper:
        def __init__(self, candidate):
            self.message = self.MessageWrapper(candidate.content)

        class MessageWrapper:
            def __init__(self, content):
                #  parts[0].text
                self.content = content.parts[0].text


