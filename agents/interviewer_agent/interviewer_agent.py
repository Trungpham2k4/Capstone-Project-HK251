from agents.base_agent.base_agent import KnowledgeDrivenAgent
from agents.interviewer_agent.profile import InterviewerProfile
from agents.interviewer_agent.thinking import InterviewerThinking
from agents.interviewer_agent.monitor import InterviewerMonitor
from agents.interviewer_agent.memory import InterviewerMemory
from agents.interviewer_agent.action import InterviewerAction
from agents.interviewer_agent.knowledge import InterviewerKnowledge
from services.kafka_service import KafkaService
from services.minio_service import MinioService


class InterviewerAgent(KnowledgeDrivenAgent):
    def __init__(self, kafka_service: KafkaService, minio_service: MinioService, llm):
        profile = InterviewerProfile()
        # knowledge = InterviewerKnowledge(host="localhost", port=6333, collection="interview_knowledge")
        # memory = InterviewerMemory(collection="interviewer_memory")

        action = InterviewerAction(publisher=kafka_service, storage_client=minio_service, llm=llm)

        thinking = InterviewerThinking(profile=profile, knowledge=None, memory=None, action=action, llm_client=llm)
        monitor = InterviewerMonitor(kafka_group_name="interviewer-group", thinking_module=thinking, kafka_service=kafka_service)

        super().__init__(
            name="Interviewer Agent",
            profile=profile,
            monitor=monitor,
            thinking=thinking,
            memory=None,
            knowledge=None,
            action=action
        )