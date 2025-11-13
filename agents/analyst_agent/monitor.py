# -------------------------
# Monitor module
# -------------------------

from typing import override
from services.kafka_service import KafkaService
from agents.enduser_agent.thinking import EndUserThinking
from agents.base_agent.monitor import MonitorModule

class AnalystMonitor(MonitorModule):
    def __init__(self, kafka_group_name: str, thinking_module: EndUserThinking, kafka_service: KafkaService):
        self.kafka_group_name = kafka_group_name
        self.thinking_module = thinking_module
        self.kafka = kafka_service
        self.topics = ["artifact_events"]
        self.pending_artifacts = {
            "user_requirements_list": None,
            "operating_environment_list": "Operating Env List.txt" # Test with pre-existing artifact
        }

        super().__init__(kafka_group_name, thinking_module, kafka_service, self.topics)

    @override
    def start(self):
        def handler(msg: dict):
            
            artifact_type = msg.get("artifact_type")

            if artifact_type not in self.pending_artifacts:
                return # Ignore irrelevant artifacts
            
            self.pending_artifacts[artifact_type] = msg.get("artifact_key")
            
            # Check if prerequisites met
            try:
                print(f"[Monitor] Received: {msg}")
                if self._all_prerequisites_met():
                    print("[Monitor] Prerequisites met, triggering Analyst...")
                    self.trigger_thinking()
            except Exception as e:
                print("[Monitor] Handler error:", e)

        self.kafka.listen(self.topics, handler, self.kafka_group_name) # Handler is on_message function

    def _all_prerequisites_met(self) -> bool:
        # Check if all required artifacts have been received
        return all(v is not None for v in self.pending_artifacts.values())
    
    @override
    def trigger_thinking(self):
        message: dict = {
            "user_requirements_list_file_name": self.pending_artifacts["user_requirements_list"],
            "operating_environment_list_file_name": self.pending_artifacts["operating_environment_list"]
        }
        self.thinking_module.decide(message)

        # Reset for next conversation
        self.pending_artifacts = {k: None for k in self.pending_artifacts}