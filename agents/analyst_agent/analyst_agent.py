from agents.base_agent.base_agent import KnowledgeDrivenAgent
from agents.analyst_agent.profile import AnalystProfile
from agents.analyst_agent.monitor import AnalystMonitor
from agents.analyst_agent.thinking import AnalystThinking
from agents.analyst_agent.memory import AnalystMemory
from agents.analyst_agent.action import AnalystAction
from agents.analyst_agent.knowledge import AnalystKnowledge
from services.kafka_service import KafkaService
from services.minio_service import MinioService

class AnalystAgent(KnowledgeDrivenAgent):
    def __init__(self, kafka_service: KafkaService, minio_service: MinioService, llm):
        profile = AnalystProfile()
        # knowledge = AnalystKnowledge(host="localhost", port=6333, collection="analyst_knowledge")
        memory = AnalystMemory()
        action = AnalystAction(publisher=kafka_service, storage_client=minio_service, profile=profile, memory=memory, llm=llm)
        thinking = AnalystThinking(profile=profile, knowledge=None, memory=memory, action=action, llm_client=llm)
        monitor = AnalystMonitor(kafka_group_name="analyst-group", thinking_module=thinking, kafka_service=kafka_service)

        

        super().__init__(
            name="Analyst Agent",
            profile=profile,
            monitor=monitor,
            thinking=thinking,
            memory=None,
            knowledge=None,
            action=action
        )