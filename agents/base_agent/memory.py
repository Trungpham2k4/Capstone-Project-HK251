# modules/memory_module.py
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime

class MemoryModule:
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
        self.client = QdrantClient(host=host, port=port)
        self.encoder = SentenceTransformer(embedding_model)
        self.collection = collection
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        
        # Ensure collection exists
        self._ensure_collection()
        
        print(f"[MemoryModule] Initialized with collection: {collection}")
    
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
                print(f"[MemoryModule] Created collection: {self.collection}")
            else:
                print(f"[MemoryModule] Collection exists: {self.collection}")
        except Exception as e:
            print(f"[MemoryModule] Error ensuring collection: {e}")
    
    def write(self, content: str, artifact_id: Optional[str] = None,
              metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Write a memory entry.
        
        Args:
            content: Memory content text
            artifact_id: Optional associated artifact ID
            metadata: Additional metadata (e.g., role, timestamp, conversation_id)
            
        Returns:
            Memory ID (UUID)
        """
        try:
            # Generate embedding
            vector = self.encoder.encode(content).tolist()
            
            # Create unique ID
            memory_id = str(uuid.uuid4())
            
            # Prepare payload
            payload = {
                "content": content,
                "artifact_id": artifact_id or "",
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": metadata or {}
            }
            
            # Create point
            point = PointStruct(
                id=memory_id,
                vector=vector,
                payload=payload
            )
            
            # Insert into Qdrant
            self.client.upsert(
                collection_name=self.collection,
                points=[point]
            )
            
            print(f"[MemoryModule] Written memory {memory_id}: {content[:50]}...")
            return memory_id
            
        except Exception as e:
            print(f"[MemoryModule] Error writing memory: {e}")
            return ""
    
    def write_batch(self, items: List[Dict[str, Any]]) -> List[str]:
        """
        Write multiple memory entries at once.
        
        Args:
            items: List of dicts with keys: content, artifact_id, metadata
            
        Returns:
            List of memory IDs
        """
        try:
            points = []
            memory_ids = []
            
            for item in items:
                content = item.get("content", "")
                if not content:
                    continue
                
                vector = self.encoder.encode(content).tolist()
                memory_id = str(uuid.uuid4())
                memory_ids.append(memory_id)
                
                payload = {
                    "content": content,
                    "artifact_id": item.get("artifact_id", ""),
                    "timestamp": datetime.utcnow().isoformat(),
                    "metadata": item.get("metadata", {})
                }
                
                point = PointStruct(
                    id=memory_id,
                    vector=vector,
                    payload=payload
                )
                points.append(point)
            
            if points:
                self.client.upsert(
                    collection_name=self.collection,
                    points=points
                )
                print(f"[MemoryModule] Written {len(points)} memories in batch")
            
            return memory_ids
            
        except Exception as e:
            print(f"[MemoryModule] Error writing batch: {e}")
            return []
    
    def semantic_search(self, query: str, top_k: int = 5,
                       conversation_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search memories semantically based on query.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            conversation_filter: Optional conversation ID to filter by
            
        Returns:
            List of dictionaries containing:
                - memory_id: Memory UUID
                - content: Memory content
                - artifact_id: Associated artifact ID
                - timestamp: When memory was created
                - score: Relevance score
                - metadata: Additional metadata
        """
        try:
            # Generate query embedding
            query_vector = self.encoder.encode(query).tolist()
            
            # Build filter if conversation specified
            search_filter = None
            if conversation_filter:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="metadata.conversation_id",
                            match=MatchValue(value=conversation_filter)
                        )
                    ]
                )
            
            # Search in Qdrant
            results = self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                limit=top_k,
                query_filter=search_filter
            )
            
            # Format results
            memories = []
            for r in results:
                payload = r.payload or {}
                memories.append({
                    "memory_id": str(r.id),
                    "content": payload.get("content", ""),
                    "artifact_id": payload.get("artifact_id", ""),
                    "timestamp": payload.get("timestamp", ""),
                    "score": r.score,
                    "metadata": payload.get("metadata", {})
                })
            
            print(f"[MemoryModule] Retrieved {len(memories)} memories for query: '{query[:50]}...'")
            return memories
            
        except Exception as e:
            print(f"[MemoryModule] Error searching memories: {e}")
            return []
    
    def get_by_artifact_id(self, artifact_id: str) -> List[Dict[str, Any]]:
        """
        Get all memories associated with a specific artifact.
        
        Args:
            artifact_id: Artifact ID to search for
            
        Returns:
            List of memory entries
        """
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            results, _ = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="artifact_id",
                            match=MatchValue(value=artifact_id)
                        )
                    ]
                ),
                limit=100
            )
            
            memories = []
            for r in results:
                payload = r.payload or {}
                memories.append({
                    "memory_id": str(r.id),
                    "content": payload.get("content", ""),
                    "artifact_id": payload.get("artifact_id", ""),
                    "timestamp": payload.get("timestamp", ""),
                    "metadata": payload.get("metadata", {})
                })
            
            print(f"[MemoryModule] Found {len(memories)} memories for artifact: {artifact_id}")
            return memories
            
        except Exception as e:
            print(f"[MemoryModule] Error getting by artifact: {e}")
            return []
    
    def get_recent_memories(self, limit: int = 10,
                           conversation_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get most recent memories.
        
        Args:
            limit: Number of memories to return
            conversation_filter: Optional conversation ID filter
            
        Returns:
            List of recent memory entries
        """
        try:
            search_filter = None
            if conversation_filter:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="metadata.conversation_id",
                            match=MatchValue(value=conversation_filter)
                        )
                    ]
                )
            
            results, _ = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=search_filter,
                limit=limit,
                order_by="timestamp"  # Note: This might not work in all Qdrant versions
            )
            
            memories = []
            for r in results:
                payload = r.payload or {}
                memories.append({
                    "memory_id": str(r.id),
                    "content": payload.get("content", ""),
                    "artifact_id": payload.get("artifact_id", ""),
                    "timestamp": payload.get("timestamp", ""),
                    "metadata": payload.get("metadata", {})
                })
            
            # Sort by timestamp (in case Qdrant didn't sort)
            memories.sort(key=lambda x: x["timestamp"], reverse=True)
            
            print(f"[MemoryModule] Retrieved {len(memories)} recent memories")
            return memories[:limit]
            
        except Exception as e:
            print(f"[MemoryModule] Error getting recent memories: {e}")
            return []
    
    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a specific memory entry.
        
        Args:
            memory_id: Memory UUID to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete(
                collection_name=self.collection,
                points_selector=[memory_id]
            )
            print(f"[MemoryModule] Deleted memory: {memory_id}")
            return True
        except Exception as e:
            print(f"[MemoryModule] Error deleting memory: {e}")
            return False
    
    def delete_by_artifact(self, artifact_id: str) -> int:
        """
        Delete all memories associated with an artifact.
        
        Args:
            artifact_id: Artifact ID
            
        Returns:
            Number of memories deleted
        """
        try:
            # First get all memories for this artifact
            memories = self.get_by_artifact_id(artifact_id)
            memory_ids = [m["memory_id"] for m in memories]
            
            if memory_ids:
                self.client.delete(
                    collection_name=self.collection,
                    points_selector=memory_ids
                )
                print(f"[MemoryModule] Deleted {len(memory_ids)} memories for artifact: {artifact_id}")
                return len(memory_ids)
            
            return 0
        except Exception as e:
            print(f"[MemoryModule] Error deleting by artifact: {e}")
            return 0
    
    def clear_conversation(self, conversation_id: str) -> int:
        """
        Clear all memories from a specific conversation.
        
        Args:
            conversation_id: Conversation ID to clear
            
        Returns:
            Number of memories deleted
        """
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            # Get all memories for this conversation
            results, _ = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.conversation_id",
                            match=MatchValue(value=conversation_id)
                        )
                    ]
                ),
                limit=1000
            )
            
            memory_ids = [str(r.id) for r in results]
            
            if memory_ids:
                self.client.delete(
                    collection_name=self.collection,
                    points_selector=memory_ids
                )
                print(f"[MemoryModule] Cleared {len(memory_ids)} memories from conversation: {conversation_id}")
                return len(memory_ids)
            
            return 0
        except Exception as e:
            print(f"[MemoryModule] Error clearing conversation: {e}")
            return 0
    
    def count_memories(self) -> int:
        """
        Count total number of memories in collection.
        
        Returns:
            Number of memories
        """
        try:
            collection_info = self.client.get_collection(self.collection)
            count = collection_info.points_count
            print(f"[MemoryModule] Total memories: {count}")
            return count
        except Exception as e:
            print(f"[MemoryModule] Error counting memories: {e}")
            return 0
    
    def update_memory(self, memory_id: str, content: str,
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update an existing memory entry.
        
        Args:
            memory_id: Memory UUID to update
            content: New content
            metadata: Updated metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate new embedding
            vector = self.encoder.encode(content).tolist()
            
            # Get existing payload
            existing = self.client.retrieve(
                collection_name=self.collection,
                ids=[memory_id]
            )
            
            if not existing:
                print(f"[MemoryModule] Memory not found: {memory_id}")
                return False
            
            old_payload = existing[0].payload or {}
            
            # Update payload
            payload = {
                "content": content,
                "artifact_id": old_payload.get("artifact_id", ""),
                "timestamp": old_payload.get("timestamp", ""),
                "metadata": metadata or old_payload.get("metadata", {})
            }
            
            # Create updated point
            point = PointStruct(
                id=memory_id,
                vector=vector,
                payload=payload
            )
            
            # Upsert (update)
            self.client.upsert(
                collection_name=self.collection,
                points=[point]
            )
            
            print(f"[MemoryModule] Updated memory: {memory_id}")
            return True
            
        except Exception as e:
            print(f"[MemoryModule] Error updating memory: {e}")
            return False