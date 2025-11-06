# agents/knowledge_agent.py
"""
Knowledge-Driven Agent base class (implements 6 modules from iReDev Sec 3.1):
 - Profile
 - Monitor
 - Thinking
 - Memory
 - Action
 - Knowledge

How to use:
 - Subclass KnowledgeDrivenAgent and override hooks (e.g., policy in thinking,
   or valid message types in should_process).
 - Provide real LLM client and publishers (KafkaProducer/MinIO client).
"""

from agents.base_agent.profile import ProfileModule
from agents.base_agent.monitor import MonitorModule
from agents.base_agent.thinking import ThinkingModule
from agents.base_agent.memory import MemoryModule
from agents.base_agent.action import ActionModule
from agents.base_agent.knowledge import KnowledgeModule


# KnowledgeDrivenAgent (orchestrator)
# -------------------------
class KnowledgeDrivenAgent:
    def __init__(self,
                 name: str,
                 profile: ProfileModule,
                 monitor: MonitorModule,
                 thinking: ThinkingModule,
                 memory: MemoryModule,
                 knowledge: KnowledgeModule,
                 action: ActionModule):
        self.name = name
        self.profile = profile
        self.knowledge = knowledge
        self.memory = memory
        self.action = action
        # monitor subscribes to topics via consumer_factory
        self.monitor = monitor
        # thinking module
        self.thinker = thinking

    def start(self):
        self.monitor.start()
        print(f"[{self.name}] agent started, monitoring topics={self.monitor.topics}")


