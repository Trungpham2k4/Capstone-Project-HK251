# -------------------------
# Monitor module
# -------------------------

from services.kafka_service import KafkaService
from agents.enduser_agent.thinking import EndUserThinking
from agents.base_agent.monitor import MonitorModule

class AnalystMonitor(MonitorModule):
    def __init__(self, kafka_group_name: str, thinking_module: EndUserThinking, kafka_service: KafkaService):
        self.kafka_group_name = kafka_group_name
        self.thinking_module = thinking_module
        self.kafka = kafka_service
        self.topics = ["interviewer_analyst", "system_requirements_generated"]

        super().__init__(kafka_group_name, thinking_module, kafka_service, self.topics)
