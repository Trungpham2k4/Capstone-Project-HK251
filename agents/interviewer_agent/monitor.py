# -------------------------
# Monitor module
# -------------------------

from services.kafka_service import KafkaService
from agents.interviewer_agent.thinking import InterviewerThinking
from agents.base_agent.monitor import MonitorModule

class InterviewerMonitor(MonitorModule):
    def __init__(self, kafka_group_name: str, thinking_module: InterviewerThinking, kafka_service: KafkaService):
        self.kafka_group_name = kafka_group_name
        self.thinking_module = thinking_module
        self.kafka = kafka_service
        self.topics = ["enduser_interviewer", "user_interviewer"]

        super().__init__(kafka_group_name, thinking_module, kafka_service, self.topics)
