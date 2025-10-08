from langchain_openai import OpenAIEmbeddings

class EmbeddingConfig:
    model_name: str
    base_url: str
    

    def __init__(self, args) -> None:
        self.model_name = args.model_embed_name

    def get_embedding_openapi(self):
        if self.model_name:
            return OpenAIEmbeddings(
                model=self.model_name
            )
        raise ValueError("Please set your model embedding name")