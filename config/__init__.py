from config.llm_config import LLMConfig
from config.embedding_config import EmbeddingConfig

class Config:
    llm_config: LLMConfig
    embedding_config: EmbeddingConfig

    @classmethod
    def get_default(cls, args):
        cls.llm_config = LLMConfig(args)
        cls.embedding_config = EmbeddingConfig(args)

    @classmethod
    def get_llm(cls):
        return cls.llm_config.get_chat_openapi()
    
    @classmethod
    def get_embedding(cls):
        return cls.embedding_config.get_embedding_openapi()