# modules/knowledge_module.py
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import uuid

from agents.base_agent.knowledge import KnowledgeModule

class InterviewerKnowledge(KnowledgeModule):
    
    def __init__(self, host: str = "localhost", port: int = 6333, 
                 collection: str = "knowledge_base",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize Knowledge Module with Qdrant.
        
        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection: Name of the collection to use
            embedding_model: Sentence transformer model for embeddings
        """
        super().__init__(host=host, port=port, collection=collection, embedding_model=embedding_model)