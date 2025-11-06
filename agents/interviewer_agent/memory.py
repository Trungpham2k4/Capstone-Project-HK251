from agents.base_agent.memory import MemoryModule

class InterviewerMemory(MemoryModule):
    """
    Memory module for storing and retrieving agent's conversation memory.
    Uses Qdrant for vector storage and retrieval.
    Supports both read and write operations.
    """
    
    def __init__(self, host: str = "localhost", port: int = 6333,
                 collection: str = "agent_memory",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize Memory Module with Qdrant.
        
        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection: Name of the collection to use
            embedding_model: Sentence transformer model for embeddings
        """
        super().__init__(host=host, port=port, collection=collection, embedding_model=embedding_model)