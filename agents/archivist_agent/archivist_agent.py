from agents.base_agent.base_agent import KnowledgeDrivenAgent
from agents.archivist_agent.profile import ArchivistProfile
from agents.archivist_agent.monitor import ArchivistMonitor
from agents.archivist_agent.thinking import ArchivistThinking
from agents.archivist_agent.memory import ArchivistMemory
from agents.archivist_agent.action import ArchivistAction
from agents.archivist_agent.knowledge import ArchivistKnowledge
from services.kafka_service import KafkaService
from services.minio_service import MinioService

class ArchivistAgent(KnowledgeDrivenAgent):
    def __init__(self, kafka_service: KafkaService, minio_service: MinioService, llm):
        profile = ArchivistProfile()
        # knowledge = ArchivistKnowledge(host="localhost", port=6333, collection="archivist_knowledge")
        memory = ArchivistMemory()
        action = ArchivistAction(publisher=kafka_service, storage_client=minio_service, profile=profile, memory=memory, llm=llm)
        thinking = ArchivistThinking(profile=profile, knowledge=None, memory=memory, action=action, llm_client=llm)
        monitor = ArchivistMonitor(kafka_group_name="archivist-group", thinking_module=thinking, kafka_service=kafka_service)

        

        super().__init__(
            name="Archivist Agent",
            profile=profile,
            monitor=monitor,
            thinking=thinking,
            memory=None,
            knowledge=None,
            action=action
        )