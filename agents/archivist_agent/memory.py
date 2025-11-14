from agents.base_agent.memory import MemoryModule
from typing import Literal, override

class ArchivistMemory(MemoryModule):
    
    def __init__(self):
        pass

    @override
    def write(self, key: str, value):
        pass
    
    @override
    def read(self, key: str):
        pass