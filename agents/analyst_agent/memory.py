from agents.base_agent.memory import MemoryModule
from typing import Literal, override


class AnalystMemory(MemoryModule):

    def __init__(self, collection: str = "agent_memory"):
        super().__init__(collection=collection)

    def get_systems_requirements(self) -> str:
        artifacts = self.get_by_artifact_id("system_requirements", 1)
        if len(artifacts) == 1:
            return artifacts[0]["content"]
        return ""

    def get_requirement_model(self) -> str:
        artifacts = self.get_by_artifact_id("requirement_model", 1)
        if len(artifacts) == 1:
            return artifacts[0]["content"]
        return ""
