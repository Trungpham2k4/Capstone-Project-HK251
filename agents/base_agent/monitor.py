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
        self.messages: dict[str, str] = {}
        self.handled_message_ids: list[str] = []

    def start(self):
        def handler(msg):
            if msg.get("message_id", None) is not None:
                if self.check_duplicate_message(msg["message_id"], self.handled_message_ids):
                    print("[Monitor] Duplicate message received, ignoring.")
                    return
                self.handled_message_ids.append(msg["message_id"])
            try:
                print(f"[Monitor] Received: {msg}")
                self.messages = msg
                self.trigger_thinking() 
            except Exception as e:
                print("[Monitor] Handler error:", e)

        self.kafka.listen(self.topics, handler, self.kafka_group_name) # Handler is on_message function

    def trigger_thinking(self):
        self.thinking_module.decide(self.messages)

    def check_duplicate_message(self, message_id: str, handled_messages: list[str]) -> bool:
        return message_id in handled_messages
