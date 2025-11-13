from agents.base_agent.knowledge import KnowledgeModule

class AnalystKnowledge(KnowledgeModule):
    
    def __init__(self, host: str = "localhost", port: int = 6333, 
                 collection: str = "knowledge_base",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__(host=host, port=port, collection=collection, embedding_model=embedding_model)