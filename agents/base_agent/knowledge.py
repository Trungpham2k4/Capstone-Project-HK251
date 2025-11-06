# modules/knowledge_module.py
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import uuid

class KnowledgeModule:
    """
    Knowledge module for storing and retrieving domain knowledge.
    Uses Qdrant for vector storage and retrieval.
    Read-only: Only supports retrieve operations.
    """
    
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
        self.client = QdrantClient(host=host, port=port)
        self.encoder = SentenceTransformer(embedding_model)
        self.collection = collection
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        
        # Ensure collection exists
        self._ensure_collection()
        
        print(f"[KnowledgeModule] Initialized with collection: {collection}")
    
    def _ensure_collection(self):
        """Ensure the collection exists, create if not."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                print(f"[KnowledgeModule] Created collection: {self.collection}")
            else:
                print(f"[KnowledgeModule] Collection exists: {self.collection}")
        except Exception as e:
            print(f"[KnowledgeModule] Error ensuring collection: {e}")
    
    def retrieve(self, query: str, k: int = 5, 
                 category_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant knowledge snippets based on query.
        
        Args:
            query: Search query text
            k: Number of results to return
            category_filter: Optional category to filter by (e.g., "requirements", "domain")
            
        Returns:
            List of dictionaries containing:
                - text: The knowledge text
                - category: Knowledge category
                - score: Relevance score
                - metadata: Additional metadata
        """
        try:
            # Generate query embedding
            query_vector = self.encoder.encode(query).tolist()
            
            # Build filter if category specified
            search_filter = None
            if category_filter:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="category",
                            match=MatchValue(value=category_filter)
                        )
                    ]
                )
            
            # Search in Qdrant
            results = self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                limit=k,
                query_filter=search_filter
            )
            
            # Format results
            snippets = []
            for r in results:
                payload = r.payload or {}
                snippets.append({
                    "text": payload.get("text", ""),
                    "category": payload.get("category", "general"),
                    "score": r.score,
                    "metadata": payload.get("metadata", {})
                })
            
            print(f"[KnowledgeModule] Retrieved {len(snippets)} snippets for query: '{query[:50]}...'")
            return snippets
            
        except Exception as e:
            print(f"[KnowledgeModule] Error retrieving knowledge: {e}")
            return []
    
    def get_by_category(self, category: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get all knowledge items in a specific category.
        
        Args:
            category: Category name
            limit: Maximum number of items to return
            
        Returns:
            List of knowledge items
        """
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            # Scroll through collection with filter
            results, _ = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="category",
                            match=MatchValue(value=category)
                        )
                    ]
                ),
                limit=limit
            )
            
            items = []
            for r in results:
                payload = r.payload or {}
                items.append({
                    "text": payload.get("text", ""),
                    "category": payload.get("category", ""),
                    "metadata": payload.get("metadata", {})
                })
            
            print(f"[KnowledgeModule] Retrieved {len(items)} items from category: {category}")
            return items
            
        except Exception as e:
            print(f"[KnowledgeModule] Error getting by category: {e}")
            return []
    
    def list_categories(self) -> List[str]:
        """
        List all available categories in the knowledge base.
        
        Returns:
            List of category names
        """
        try:
            # Scroll through all points and extract unique categories
            results, _ = self.client.scroll(
                collection_name=self.collection,
                limit=1000  # Adjust based on your needs
            )
            
            categories = set()
            for r in results:
                payload = r.payload or {}
                category = payload.get("category", "general")
                categories.add(category)
            
            category_list = sorted(list(categories))
            print(f"[KnowledgeModule] Found categories: {category_list}")
            return category_list
            
        except Exception as e:
            print(f"[KnowledgeModule] Error listing categories: {e}")
            return []
    
    def count_knowledge_items(self) -> int:
        """
        Count total number of knowledge items in collection.
        
        Returns:
            Number of items
        """
        try:
            collection_info = self.client.get_collection(self.collection)
            count = collection_info.points_count
            print(f"[KnowledgeModule] Total knowledge items: {count}")
            return count
        except Exception as e:
            print(f"[KnowledgeModule] Error counting items: {e}")
            return 0
    
    # Admin methods (optional, for populating knowledge base)
    def _add_knowledge(self, text: str, category: str = "general", 
                       metadata: Optional[Dict[str, Any]] = None):
        """
        Internal method to add knowledge to the base.
        Should be used by admin tools, not during normal operation.
        
        Args:
            text: Knowledge text
            category: Category classification
            metadata: Additional metadata
        """
        try:
            vector = self.encoder.encode(text).tolist()
            
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": text,
                    "category": category,
                    "metadata": metadata or {}
                }
            )
            
            self.client.upsert(
                collection_name=self.collection,
                points=[point]
            )
            
            print(f"[KnowledgeModule] Added knowledge: {text[:50]}...")
            
        except Exception as e:
            print(f"[KnowledgeModule] Error adding knowledge: {e}")
    
    def _bulk_add_knowledge(self, items: List[Dict[str, Any]]):
        """
        Bulk add multiple knowledge items.
        
        Args:
            items: List of dicts with keys: text, category, metadata
        """
        try:
            points = []
            for item in items:
                text = item.get("text", "")
                if not text:
                    continue
                
                vector = self.encoder.encode(text).tolist()
                
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "text": text,
                        "category": item.get("category", "general"),
                        "metadata": item.get("metadata", {})
                    }
                )
                points.append(point)
            
            if points:
                self.client.upsert(
                    collection_name=self.collection,
                    points=points
                )
                print(f"[KnowledgeModule] Bulk added {len(points)} knowledge items")
            
        except Exception as e:
            print(f"[KnowledgeModule] Error bulk adding knowledge: {e}")


# Example usage and helper functions
def populate_sample_knowledge(knowledge_module: KnowledgeModule):
    """
    Populate knowledge base with sample requirements engineering knowledge.
    """
    sample_knowledge = [
        {
            "text": "Requirements elicitation involves gathering requirements from stakeholders through interviews, workshops, and observations.",
            "category": "requirements_elicitation",
            "metadata": {"source": "RE textbook", "topic": "elicitation_methods"}
        },
        {
            "text": "Functional requirements describe what the system should do, while non-functional requirements describe how the system should perform.",
            "category": "requirements_types",
            "metadata": {"source": "IEEE 830", "topic": "classification"}
        },
        {
            "text": "Use the 5W1H method (Who, What, When, Where, Why, How) to explore requirements comprehensively.",
            "category": "elicitation_techniques",
            "metadata": {"source": "Best practices", "topic": "questioning"}
        },
        {
            "text": "Requirements should be clear, unambiguous, verifiable, and traceable.",
            "category": "requirements_quality",
            "metadata": {"source": "ISO 29148", "topic": "quality_attributes"}
        },
        {
            "text": "Open-ended questions encourage detailed responses: 'Can you describe your current process?' vs 'Is your process good?'",
            "category": "elicitation_techniques",
            "metadata": {"source": "Interview guide", "topic": "questioning"}
        }
    ]
    
    knowledge_module._bulk_add_knowledge(sample_knowledge)
    print("[KnowledgeModule] Sample knowledge populated")