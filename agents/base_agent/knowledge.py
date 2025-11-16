# modules/knowledge_module.py
from dotenv import load_dotenv

load_dotenv()

import os
import time
from pathlib import Path
from typing import Dict, List, Optional
from enum import Enum
from datetime import datetime
import json
import hashlib

from pydantic import BaseModel, Field
from watchdog.observers import Observer
from watchdog.events import (
    FileSystemEventHandler,
    FileCreatedEvent,
    FileModifiedEvent,
    FileDeletedEvent,
    FileMovedEvent
)

from docling.document_converter import DocumentConverter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import OpenAI

# ===================== CONSTANTS & CONFIGURATION =====================

# Qdrant configuration
QDRANT_URL = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "knowledge_collection"
VECTOR_SIZE = 1536
VECTOR_DISTANCE = Distance.COSINE

# OpenAI configuration
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_MAX_CHARS = 8000

# File system configuration
WATCH_DIRECTORY = "knowledge_storage"
PROCESSED_FILES_CACHE = "processed_files.json"
MIN_CONTENT_LENGTH = 50
FILE_WRITE_DELAY = 2
FILE_MODIFY_DELAY = 1

# Supported file extensions (docling support)
SUPPORTED_EXTENSIONS = {
    # Documents
    '.pdf', '.docx', '.doc', '.html', '.htm', '.md', '.txt', '.rtf', '.adoc',
    # Office formats
    '.pptx', '.xlsx', '.xls',
    # Images (OCR)
    '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif',
    # XML formats
    '.xml', '.jats', '.nxml',
    # E-book
    '.epub',
    # Audio (ASR)
    '.wav', '.mp3', '.m4a'
}

# Folder to knowledge configuration mapping
FOLDER_TO_CONFIG: Dict[str, Dict] = {
    # Domain Knowledge
    "domain_terminology_glossary": {
        "type": "Domain",
        "phases": ["Elicitation", "Analysis", "Specification", "Validation"],
        "agents": ["Interviewer", "Analyst", "Archivist", "Reviewer"]
    },
    "industry_processes_regulations": {
        "type": "Domain",
        "phases": ["Elicitation", "Analysis", "Validation"],
        "agents": ["Interviewer", "Analyst", "Reviewer"]
    },
    "nasa_faa_vv_cases": {
        "type": "Domain",
        "phases": ["Elicitation", "Analysis"],
        "agents": ["Interviewer", "Analyst"]
    },

    # Typical Methodology
    "interviews_workshops": {
        "type": "Methodology",
        "phases": ["Elicitation"],
        "agents": ["Interviewer", "Enduser", "Deployer"]
    },
    "uml_sysml_modeling": {
        "type": "Methodology",
        "phases": ["Analysis"],
        "agents": ["Analyst"]
    },
    "sysml_mbse_modeling": {
        "type": "Methodology",
        "phases": ["Analysis"],
        "agents": ["Analyst"]
    },
    "bpmn_modeling": {
        "type": "Methodology",
        "phases": ["Analysis"],
        "agents": ["Analyst"]
    },
    "behavior_driven_specification": {
        "type": "Methodology",
        "phases": ["Specification", "Validation"],
        "agents": ["Archivist", "Reviewer"]
    },
    "formal_specification": {
        "type": "Methodology",
        "phases": ["Specification", "Validation"],
        "agents": ["Archivist", "Reviewer"]
    },
    "inspection_peer_review": {
        "type": "Methodology",
        "phases": ["Validation"],
        "agents": ["Reviewer"]
    },
    "formal_validation": {
        "type": "Methodology",
        "phases": ["Validation"],
        "agents": ["Reviewer"]
    },

    # Standards
    "iso_iec_ieee_29148": {
        "type": "Standard",
        "phases": ["Elicitation", "Analysis"],
        "agents": ["Interviewer", "Analyst", "Archivist"]
    },
    "iso_iec_24744": {
        "type": "Standard",
        "phases": ["Analysis"],
        "agents": ["Analyst"]
    },
    "bpmn_2_0": {
        "type": "Standard",
        "phases": ["Analysis"],
        "agents": ["Analyst"]
    },
    "ieee_1012_2016": {
        "type": "Standard",
        "phases": ["Validation"],
        "agents": ["Reviewer"]
    },
    "iso_26262_6": {
        "type": "Standard",
        "phases": ["Validation"],
        "agents": ["Reviewer"]
    },

    # Artifacts Template
    "ieee_830_srs_template": {
        "type": "Template",
        "phases": ["Specification", "Validation"],
        "agents": ["Archivist", "Reviewer"]
    },
    "use_case_specification": {
        "type": "Template",
        "phases": ["Elicitation", "Analysis"],
        "agents": ["Interviewer", "Analyst"]
    },
    "reqif_based_specification": {
        "type": "Template",
        "phases": ["Specification", "Validation"],
        "agents": ["Archivist", "Reviewer"]
    },
    "vv_plan_outline": {
        "type": "Template",
        "phases": ["Validation"],
        "agents": ["Reviewer"]
    },
    "requirements_traceability_matrix": {
        "type": "Template",
        "phases": ["Validation"],
        "agents": ["Reviewer", "Archivist"]
    },
    "review_checklists": {
        "type": "Template",
        "phases": ["Validation"],
        "agents": ["Reviewer"]
    },

    # Common Strategies
    "5w1h": {
        "type": "Strategy",
        "phases": ["Elicitation"],
        "agents": ["Interviewer"]
    },
    "moscow": {
        "type": "Strategy",
        "phases": ["Analysis"],
        "agents": ["Analyst"]
    },
    "socratic_questioning": {
        "type": "Strategy",
        "phases": ["Elicitation"],
        "agents": ["Interviewer"]
    },
    "requirements_tradeoff": {
        "type": "Strategy",
        "phases": ["Specification", "Validation"],
        "agents": ["Analyst", "Reviewer"]
    },
}

# Category folders mapping for folder structure creation
CATEGORY_FOLDERS = {
    "domain_knowledge": ["domain_terminology_glossary", "industry_processes_regulations", "nasa_faa_vv_cases"],
    "typical_methodology": ["interviews_workshops", "uml_sysml_modeling", "sysml_mbse_modeling",
                            "bpmn_modeling", "behavior_driven_specification", "formal_specification",
                            "inspection_peer_review", "formal_validation"],
    "standards": ["iso_iec_ieee_29148", "iso_iec_24744", "bpmn_2_0", "ieee_1012_2016", "iso_26262_6"],
    "artifacts_template": ["ieee_830_srs_template", "use_case_specification", "reqif_based_specification",
                           "vv_plan_outline", "requirements_traceability_matrix", "review_checklists"],
    "common_strategies": ["5w1h", "moscow", "socratic_questioning", "requirements_tradeoff"]
}


# ===================== ENUMS & MODELS =====================

class KnowledgeType(str, Enum):
    DOMAIN = "Domain"
    METHODOLOGY = "Methodology"
    STANDARD = "Standard"
    TEMPLATE = "Template"
    STRATEGY = "Strategy"


class KnowledgeSource(str, Enum):
    LITERATURE = "Literature"
    PROJECT = "Project"
    EXPERT = "Expert"


class LifecyclePhase(str, Enum):
    ELICITATION = "Elicitation"
    ANALYSIS = "Analysis"
    SPECIFICATION = "Specification"
    VALIDATION = "Validation"


class AgentType(str, Enum):
    INTERVIEWER = "Interviewer"
    ENDUSER = "Enduser"
    DEPLOYER = "Deployer"
    ANALYST = "Analyst"
    ARCHIVIST = "Archivist"
    REVIEWER = "Reviewer"


class KnowledgeOntology(BaseModel):
    """Knowledge payload for Qdrant"""
    type: KnowledgeType
    source: KnowledgeSource
    name: str
    content: str
    lifecycle_phase: List[LifecyclePhase]
    applicable_agents: List[AgentType]
    file_path: str
    file_hash: str
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# ===================== KNOWLEDGE VECTORIZER =====================

class KnowledgeVectorizer:
    def __init__(
            self,
            qdrant_url: str = QDRANT_URL,
            qdrant_port: int = QDRANT_PORT,
            collection_name: str = COLLECTION_NAME,
            openai_api_key: Optional[str] = None,
            embedding_model: str = EMBEDDING_MODEL
    ):
        self.qdrant_client = QdrantClient(url=qdrant_url, port=qdrant_port)
        self.collection_name = collection_name
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.embedding_model = embedding_model
        self.doc_converter = DocumentConverter()

        self._ensure_collection()
        self.processed_files = self._load_processed_files()

    def _ensure_collection(self):
        """Create Qdrant collection if it doesn't exist"""
        collections = self.qdrant_client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if not exists:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=VECTOR_DISTANCE)
            )
            print(f"Created collection: {self.collection_name}")

    def _load_processed_files(self) -> Dict[str, str]:
        """Load record of processed files"""
        if os.path.exists(PROCESSED_FILES_CACHE):
            with open(PROCESSED_FILES_CACHE, 'r') as f:
                return json.load(f)
        return {}

    def _save_processed_files(self):
        """Save record of processed files"""
        with open(PROCESSED_FILES_CACHE, 'w') as f:
            json.dump(self.processed_files, f, indent=2)

    def _compute_file_hash(self, file_path: str) -> str:
        """Compute MD5 hash of file"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            hasher.update(f.read())
        return hasher.hexdigest()

    def _get_point_id(self, file_path: str) -> int:
        """Generate stable point ID from file path"""
        return abs(hash(file_path)) % (10 ** 8)

    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI"""
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding

    def _extract_text_from_file(self, file_path: str) -> str:
        """Extract text from document using Docling"""
        try:
            result = self.doc_converter.convert(file_path)
            markdown_text = result.document.export_to_markdown()
            return markdown_text
        except Exception as e:
            print(f"Error extracting text from {file_path}: {e}")
            return ""

    def _determine_knowledge_metadata(self, file_path: str) -> Optional[Dict]:
        """Determine knowledge type, phases, and agents from folder structure"""
        path_parts = Path(file_path).parts

        for folder_name, config in FOLDER_TO_CONFIG.items():
            if folder_name in path_parts:
                return {
                    "type": config["type"],
                    "phases": config["phases"],
                    "agents": config["agents"]
                }

        return None

    def _infer_source(self, file_name: str) -> KnowledgeSource:
        """Infer knowledge source from file name (basic heuristic)"""
        name_lower = file_name.lower()

        if any(x in name_lower for x in ['ieee', 'iso', 'standard', 'spec']):
            return KnowledgeSource.LITERATURE
        elif any(x in name_lower for x in ['project', 'case', 'example']):
            return KnowledgeSource.PROJECT
        elif any(x in name_lower for x in ['expert', 'interview', 'practice']):
            return KnowledgeSource.EXPERT

        return KnowledgeSource.LITERATURE

    def add_or_update_file(self, file_path: str) -> bool:
        """Add or update a file in the knowledge base"""
        print(f"Processing: {file_path}")

        content = self._extract_text_from_file(file_path)
        if not content or len(content.strip()) < MIN_CONTENT_LENGTH:
            print(f"Insufficient content in {file_path}")
            return False

        metadata = self._determine_knowledge_metadata(file_path)
        if not metadata:
            print(f"Cannot determine knowledge configuration for {file_path}")
            return False

        k_type = metadata["type"]
        lifecycle_phases = metadata["phases"]
        agents = metadata["agents"]

        file_name = Path(file_path).stem
        source = self._infer_source(file_name)
        file_hash = self._compute_file_hash(file_path)

        is_update = file_path in self.processed_files

        ontology = KnowledgeOntology(
            type=k_type,
            source=source,
            name=file_name,
            content=content,
            lifecycle_phase=lifecycle_phases,
            applicable_agents=agents,
            file_path=file_path,
            file_hash=file_hash,
            updated_at=datetime.now().isoformat()
        )

        try:
            embedding = self._get_embedding(content[:EMBEDDING_MAX_CHARS])
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return False

        try:
            point_id = self._get_point_id(file_path)

            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=ontology.model_dump()
                    )
                ]
            )

            self.processed_files[file_path] = file_hash
            self._save_processed_files()

            action = "Updated" if is_update else "Added"
            print(f"{action}: {file_path}")
            print(f"  - Type: {k_type}")
            print(f"  - Phases: {lifecycle_phases}")
            print(f"  - Agents: {agents}")
            return True

        except Exception as e:
            print(f"Error upserting to Qdrant: {e}")
            return False

    def delete_file(self, file_path: str) -> bool:
        """Delete a file from the knowledge base"""
        try:
            point_id = self._get_point_id(file_path)

            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=[point_id]
            )

            if file_path in self.processed_files:
                del self.processed_files[file_path]
                self._save_processed_files()

            print(f"Deleted: {file_path}")
            return True

        except Exception as e:
            print(f"Error deleting from Qdrant: {e}")
            return False


# ===================== FILE WATCHER =====================

class KnowledgeFileHandler(FileSystemEventHandler):
    def __init__(self, vectorizer: KnowledgeVectorizer):
        self.vectorizer = vectorizer
        super().__init__()

    def _is_supported_file(self, file_path: str) -> bool:
        """Check if file is supported"""
        ext = Path(file_path).suffix.lower()
        return ext in SUPPORTED_EXTENSIONS

    def on_created(self, event):
        """Handle file creation"""
        if event.is_directory:
            return

        file_path = event.src_path
        if self._is_supported_file(file_path):
            print(f"\nNew file detected: {file_path}")
            time.sleep(FILE_WRITE_DELAY)
            self.vectorizer.add_or_update_file(file_path)

    def on_modified(self, event):
        """Handle file modification"""
        if event.is_directory:
            return

        file_path = event.src_path
        if self._is_supported_file(file_path):
            if file_path in self.vectorizer.processed_files:
                try:
                    new_hash = self.vectorizer._compute_file_hash(file_path)
                    old_hash = self.vectorizer.processed_files[file_path]

                    if new_hash != old_hash:
                        print(f"\nFile modified: {file_path}")
                        time.sleep(FILE_MODIFY_DELAY)
                        self.vectorizer.add_or_update_file(file_path)
                except Exception as e:
                    print(f"Error checking file modification: {e}")

    def on_deleted(self, event):
        """Handle file deletion"""
        if event.is_directory:
            return

        file_path = event.src_path
        if self._is_supported_file(file_path):
            print(f"\nFile deleted: {file_path}")
            self.vectorizer.delete_file(file_path)

    def on_moved(self, event):
        """Handle file move/rename"""
        if event.is_directory:
            return

        src_path = event.src_path
        dest_path = event.dest_path

        if self._is_supported_file(src_path):
            print(f"\nFile moved: {src_path} -> {dest_path}")
            self.vectorizer.delete_file(src_path)

            if self._is_supported_file(dest_path):
                time.sleep(FILE_MODIFY_DELAY)
                self.vectorizer.add_or_update_file(dest_path)


# ===================== KNOWLEDGE MODULE =====================

class KnowledgeModule:
    """Knowledge module for agent system - handles knowledge storage and retrieval"""

    def __init__(
            self,
            watch_directory: str = WATCH_DIRECTORY,
            qdrant_url: str = QDRANT_URL,
            qdrant_port: int = QDRANT_PORT,
            openai_api_key: Optional[str] = None,
            auto_start: bool = False
    ):
        self.watch_directory = watch_directory
        self.vectorizer = KnowledgeVectorizer(
            qdrant_url=qdrant_url,
            qdrant_port=qdrant_port,
            openai_api_key=openai_api_key
        )
        self.observer = Observer()
        self.is_watching = False

        if auto_start:
            self.initialize()

    def initialize(self):
        """Initialize the knowledge module"""
        print("\n" + "=" * 60)
        print("KNOWLEDGE MODULE INITIALIZATION")
        print("=" * 60)

        self._create_folder_structure()
        self._process_existing_files()
        self._print_status()

    def _create_folder_structure(self):
        """Create folder structure if it doesn't exist"""
        print("\nCreating folder structure...")

        base_path = Path(self.watch_directory)
        base_path.mkdir(exist_ok=True)

        created_count = 0

        for category, subfolders in CATEGORY_FOLDERS.items():
            category_path = base_path / category

            if not category_path.exists():
                category_path.mkdir(parents=True, exist_ok=True)
                created_count += 1
                print(f"  Created: {category}/")

            for subfolder in subfolders:
                subfolder_path = category_path / subfolder
                if not subfolder_path.exists():
                    subfolder_path.mkdir(parents=True, exist_ok=True)
                    created_count += 1
                    print(f"  Created: {category}/{subfolder}")

        if created_count == 0:
            print("  All folders already exist")
        else:
            print(f"  Created {created_count} folders")

    def _process_existing_files(self):
        """Process all existing files in the directory"""
        print(f"\nScanning existing files...")

        processed = 0
        skipped = 0

        for root, _, files in os.walk(self.watch_directory):
            for file in files:
                file_path = os.path.join(root, file)
                ext = Path(file_path).suffix.lower()

                if ext in SUPPORTED_EXTENSIONS:
                    if file_path in self.vectorizer.processed_files:
                        try:
                            current_hash = self.vectorizer._compute_file_hash(file_path)
                            if current_hash == self.vectorizer.processed_files[file_path]:
                                skipped += 1
                                continue
                        except:
                            pass

                    if self.vectorizer.add_or_update_file(file_path):
                        processed += 1

        print(f"\nProcessed: {processed} files")
        print(f"Skipped: {skipped} files (already up-to-date)")

    def _print_status(self):
        """Print current knowledge base status"""
        try:
            collection_info = self.vectorizer.qdrant_client.get_collection(
                self.vectorizer.collection_name
            )
            total = collection_info.points_count

            print("\n" + "=" * 60)
            print("KNOWLEDGE BASE STATUS")
            print("=" * 60)
            print(f"Total documents: {total}")

            for k_type in KnowledgeType:
                results = self.vectorizer.qdrant_client.scroll(
                    collection_name=self.vectorizer.collection_name,
                    scroll_filter={
                        "must": [{"key": "type", "match": {"value": k_type.value}}]
                    },
                    limit=1000
                )
                count = len(results[0])
                if count > 0:
                    print(f"  {k_type.value}: {count}")

            print("=" * 60)

        except Exception as e:
            print(f"Error getting status: {e}")

    def get_status(self) -> Dict:
        """Get current knowledge base status as dictionary"""
        try:
            collection_info = self.vectorizer.qdrant_client.get_collection(
                self.vectorizer.collection_name
            )
            total = collection_info.points_count

            type_counts = {}
            for k_type in KnowledgeType:
                results = self.vectorizer.qdrant_client.scroll(
                    collection_name=self.vectorizer.collection_name,
                    scroll_filter={
                        "must": [{"key": "type", "match": {"value": k_type.value}}]
                    },
                    limit=1000
                )
                count = len(results[0])
                if count > 0:
                    type_counts[k_type.value] = count

            return {
                "total_documents": total,
                "by_type": type_counts,
                "is_watching": self.is_watching,
                "watch_directory": self.watch_directory
            }

        except Exception as e:
            return {"error": str(e)}

    def start_watching(self):
        """Start watching for file changes"""
        if self.is_watching:
            print("Already watching for file changes")
            return

        event_handler = KnowledgeFileHandler(self.vectorizer)
        self.observer.schedule(event_handler, self.watch_directory, recursive=True)
        self.observer.start()
        self.is_watching = True

        print(f"\nWatching: {self.watch_directory}")
        print("Monitoring all file changes (create, modify, delete, move)")

    def stop_watching(self):
        """Stop watching for file changes"""
        if not self.is_watching:
            print("Not currently watching")
            return

        self.observer.stop()
        self.observer.join()
        self.is_watching = False
        print("\nStopped watching")
        self._print_status()

    def search_knowledge(
            self,
            query: str,
            k_type: Optional[KnowledgeType] = None,
            lifecycle_phase: Optional[LifecyclePhase] = None,
            agent_type: Optional[AgentType] = None,
            limit: int = 5
    ) -> List[Dict]:
        """Search knowledge base with filters"""
        try:
            # Generate query embedding
            query_vector = self.vectorizer._get_embedding(query)

            # Build filter
            filter_conditions = []
            if k_type:
                filter_conditions.append({"key": "type", "match": {"value": k_type.value}})
            if lifecycle_phase:
                filter_conditions.append({"key": "lifecycle_phase", "match": {"any": [lifecycle_phase.value]}})
            if agent_type:
                filter_conditions.append({"key": "applicable_agents", "match": {"any": [agent_type.value]}})

            search_filter = {"must": filter_conditions} if filter_conditions else None

            # Search
            results = self.vectorizer.qdrant_client.search(
                collection_name=self.vectorizer.collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=search_filter
            )

            return [
                {
                    "score": result.score,
                    "name": result.payload.get("name"),
                    "type": result.payload.get("type"),
                    "content": result.payload.get("content")[:500] + "...",  # Preview
                    "file_path": result.payload.get("file_path"),
                    "lifecycle_phase": result.payload.get("lifecycle_phase"),
                    "applicable_agents": result.payload.get("applicable_agents")
                }
                for result in results
            ]

        except Exception as e:
            print(f"Error searching knowledge: {e}")
            return []

    def add_file(self, file_path: str) -> bool:
        """Manually add a file to knowledge base"""
        return self.vectorizer.add_or_update_file(file_path)

    def remove_file(self, file_path: str) -> bool:
        """Manually remove a file from knowledge base"""
        return self.vectorizer.delete_file(file_path)

    def run(self):
        """Run the knowledge module with file watching (blocking)"""
        self.initialize()
        self.start_watching()

        print("Press Ctrl+C to stop...\n")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop_watching()


# ===================== EXAMPLE USAGE =====================

if __name__ == "__main__":
    # Initialize knowledge module
    knowledge_module = KnowledgeModule(
        watch_directory=WATCH_DIRECTORY,
        qdrant_url=QDRANT_URL,
        qdrant_port=QDRANT_PORT,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Run with file watching
    knowledge_module.run()