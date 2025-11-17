from agents.base_agent.memory import MemoryModule


class InterviewerMemory(MemoryModule):
    """
    Memory module for storing and retrieving agent's conversation memory.
    Uses Qdrant for vector storage and retrieval.
    Supports both read and write operations.
    """

    def __init__(self, collection: str = "agent_memory"):
        """
        Initialize Memory Module with Qdrant.

        Args:
            collection: Name of the collection to use
        """
        super().__init__(collection=collection)
