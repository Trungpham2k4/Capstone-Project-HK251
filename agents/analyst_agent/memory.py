from agents.base_agent.memory import MemoryModule
from typing import Literal, override

class AnalystMemory(MemoryModule):
    
    def __init__(self):
        self.generated_system_requirements: bool = False
        self.requirement_model_chosen: bool = False
        self.system_requirements_content: str = ""
        self.requirement_model_type: Literal["Use case diagram", "SysML diagram"] = None

    @override
    def write(self, key: str, value):
        if key == "system_requirements":
            self.system_requirements_content = value
            self.generated_system_requirements = True
        elif key == "requirement_model":
            self.requirement_model_type = value
            self.requirement_model_chosen = True
    
    @override
    def read(self, key: str):
        if key == "system_requirements":
            return self.system_requirements_content, self.generated_system_requirements
        elif key == "requirement_model":
            return self.requirement_model_type, self.requirement_model_chosen
        return None