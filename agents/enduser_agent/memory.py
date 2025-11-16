from agents.base_agent.memory import MemoryModule


class EndUserMemory(MemoryModule):

    def __init__(self, collection: str = "agent_memory"):
        super().__init__(collection=collection)
