# -------------------------
# Monitor module
# -------------------------

from services.kafka_service import KafkaService
from agents.base_agent.thinking import ThinkingModule

class MonitorModule:
    def __init__(self, kafka_group_name: str, thinking_module: ThinkingModule, kafka_service: KafkaService, subscribe_topics: list[str]):
        self.kafka_group_name = kafka_group_name
        self.thinking_module = thinking_module
        self.kafka = kafka_service
        self.topics = subscribe_topics
        self.on_artifact_callback = None
        self.messages: dict[str, str] = {}

    def start(self):
        def handler(msg):
            try:
                print(f"[Monitor] Received: {msg}")
                self.messages = msg
                self.trigger_thinking() 
            except Exception as e:
                print("[Monitor] Handler error:", e)

        self.kafka.listen(self.topics, handler, self.kafka_group_name) # Handler is on_message function

    def trigger_thinking(self):
        self.thinking_module.decide(self.messages)
