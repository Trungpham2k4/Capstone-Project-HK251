from utils.common import load_params


class Config:

    param_config: dict

    @classmethod
    def load_config(cls, file_path="config/config.yml"):
        cls.param_config = load_params(file_path)

    @classmethod
    def get_kafka_brokers(cls):
        try:
            return cls.param_config["kafka"]["brokers"]
        except Exception:
            return ["localhost:9092"]

    @classmethod
    def get_minio_endpoint(cls):
        try:
            return cls.param_config["minio"]["endpoint"]
        except Exception:
            return "localhost:9000"

    @classmethod
    def get_openai_base_url(cls):
        try:
            return cls.param_config["llm"]["base_url"]
        except Exception:
            return None

    @classmethod
    def get_llm_model_name(cls):
        try:
            return cls.param_config["llm"]["model_name"]
        except Exception:
            return "gpt-5-nano"

    @classmethod
    def get_embed_name(cls):
        try:
            return cls.param_config["llm"]["embed_name"]
        except Exception:
            return "sentence-transformers/all-MiniLM-L6-v2"

    @classmethod
    def get_embed_dim(cls):
        try:
            return cls.param_config["llm"]["embed_dim"]
        except Exception:
            return 384

    @classmethod
    def get_qdrant_host(cls):
        try:
            return cls.param_config["qdrant"]["host"]
        except Exception:
            return "localhost"

    @classmethod
    def get_qdrant_port(cls):
        try:
            return cls.param_config["qdrant"]["port"]
        except Exception:
            return 6333
