from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from config import Config

class DocumentKnowledge:
    knowledge_data = [
    # Domain Knowledge
    "Domain terminology: Glossary of software engineering terms, e.g., 'functional requirement' refers to specific system behavior.",
    "Industry regulations: ISO/IEC/IEEE 29148 specifies requirements for completeness, traceability, and consistency.",
    # Typical Methodologies
    "Elicitation methodology: Use 5W1H (Who, What, When, Where, Why, How) to structure interviews.",
    "Modeling methodology: UML use case diagrams map actors to system functions.",
    # Standards
    "Standard: ISO/IEC/IEEE 29148 ensures requirements are verifiable and traceable.",
    # Artifact Templates
    "IEEE 830 SRS template: Includes sections for Purpose, Scope, Functional Requirements, Non-Functional Requirements.",
    # Common Strategies
    "MoSCoW prioritization: Must have, Should have, Could have, Won't have for requirement prioritization.",
    ]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    @classmethod
    def get_vector_document(cls):
        documents = [Document(page_content=text, metadata={"source": "iREDev_knowledge"}) for text in  cls.knowledge_data]
        split_docs = cls.text_splitter.split_documents(documents)
        return FAISS.from_documents(split_docs, Config.get_embedding())