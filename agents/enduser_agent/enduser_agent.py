from agents.base_agent.base_agent import KnowledgeDrivenAgent
from agents.enduser_agent.profile import EndUserProfile
from agents.enduser_agent.monitor import EndUserMonitor
from agents.enduser_agent.thinking import EndUserThinking
from agents.enduser_agent.memory import EndUserMemory
from agents.enduser_agent.action import EndUserAction
from agents.enduser_agent.knowledge import EndUserKnowledge
from services.kafka_service import KafkaService
from services.minio_service import MinioService

class EndUserAgent(KnowledgeDrivenAgent):
    def __init__(self, kafka_service: KafkaService, minio_service: MinioService, llm):
        profile = EndUserProfile()
        # knowledge = EndUserKnowledge(host="localhost", port=6333, collection="enduser_knowledge")
        # memory = EndUserMemory(collection="enduser_memory")
        action = EndUserAction(publisher=kafka_service, storage_client=minio_service, llm=llm)
        thinking = EndUserThinking(profile=profile, knowledge=None, memory=None, action=action, llm_client=llm)
        monitor = EndUserMonitor(kafka_group_name="enduser-group", thinking_module=thinking, kafka_service=kafka_service)

        

        super().__init__(
            name="EndUser Agent",
            profile=profile,
            monitor=monitor,
            thinking=thinking,
            memory=None,
            knowledge=None,
            action=action
        )