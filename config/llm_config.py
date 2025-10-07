from langchain_openai import ChatOpenAI

class LLMConfig:
    model_name: str
    base_url: str
    temperature: float

    def __init__(self, args):
        self.model_name = args.model_name
        self.base_url = args.model_base_url
        self.temperature = args.model_temperature

    def get_chat_openapi(self):
        if self.model_name:
            return ChatOpenAI(
                base_url=self.base_url,
                model=self.model_name,
                temperature=self.temperature
            )
        raise ValueError("Please set your model name")