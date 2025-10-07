import os, json

from langchain.agents import Tool
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from rag import Rag


ARTIFACT_DIR = "artifact_pool"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# -------------------
# Build retriever + tools
# -------------------
# emb = OpenAIEmbeddings(model="text-embedding-3-small")
# vs = FAISS.from_texts([
#     "Payment must follow PCI DSS; workflow: auth -> capture -> settle.",
#     "Deployment env: 2 web servers, 1 DB cluster, TLS, port 443.",
#     "Security: must support OAuth2, RBAC, password min length 8."
# ], emb)
# retriever = vs.as_retriever(search_kwargs={"k": 2})

def knowledge_retriever_tool(query: str) -> str:
    return Rag.get_rag_chain().invoke(input={"input": query})["answer"]

class ArtifactWriterInput(BaseModel):
    data: str = Field(..., description="The content to write to the file.")
    filename: str = Field(..., description="The name of the file to write the content to.")

class ArtifactReaderInput(BaseModel):
    filename: str = Field(..., description="The name of the file to read the content from.")

def artifact_writer_tool(data: str, filename: str = "") -> str:
    print("File name:", filename)
    path = os.path.join(ARTIFACT_DIR, f"{filename}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(data)
    return f"Artifact written to {path}"

def artifact_reader_tool(filename: str = "") -> str:
    path = os.path.join(ARTIFACT_DIR, f"{filename}")
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    return data

# -------------------
# Deployment Criteria (from ISO/IEC/IEEE 29148)
# -------------------
DEPLOYMENT_CRITERIA = {
    "infrastructure": {
        "description": "Hardware infrastructure (servers, network topology, load balancers)",
        "answered": False,
        "sufficient": False
    },
    "security": {
        "description": "Security requirements (authentication, authorization, encryption, compliance)",
        "answered": False,
        "sufficient": False
    },
    "scalability": {
        "description": "Scalability and performance constraints (concurrent users, response time)",
        "answered": False,
        "sufficient": False
    },
    "database": {
        "description": "Database and data management (DBMS, backup, replication)",
        "answered": False,
        "sufficient": False
    },
    "deployment_process": {
        "description": "Deployment and CI/CD pipeline (automation, versioning, rollback)",
        "answered": False,
        "sufficient": False
    },
    "monitoring": {
        "description": "Monitoring and logging requirements (metrics, alerts, log retention)",
        "answered": False,
        "sufficient": False
    },
    "compliance": {
        "description": "Regulatory and compliance requirements (GDPR, HIPAA, industry standards)",
        "answered": False,
        "sufficient": False
    }
}

# -------------------
# Criteria Evaluation Tool
# -------------------
class CriteriaEvaluationInput(BaseModel):
    criteria_key: str = Field(..., description="The key of the criteria being evaluated")
    is_answered: bool = Field(..., description="Whether the deployer provided relevant information")
    is_sufficient: bool = Field(..., description="Whether the information is sufficient and complete")
    reasoning: str = Field(..., description="Brief explanation of the evaluation")

def evaluate_criteria_tool(
        criteria_key: str,
        is_answered: bool,
        is_sufficient: bool,
        reasoning: str
) -> str:
    """Tool for interviewer to evaluate if a criteria has been adequately addressed."""
    if criteria_key not in DEPLOYMENT_CRITERIA:
        return f"Error: Invalid criteria key '{criteria_key}'"

    DEPLOYMENT_CRITERIA[criteria_key]["answered"] = is_answered
    DEPLOYMENT_CRITERIA[criteria_key]["sufficient"] = is_sufficient

    # Save updated criteria state
    criteria_path = os.path.join(ARTIFACT_DIR, "deployment_criteria_state.json")
    with open(criteria_path, "w", encoding="utf-8") as f:
        json.dump(DEPLOYMENT_CRITERIA, f, indent=2)

    return f"Criteria '{criteria_key}' evaluated: answered={is_answered}, sufficient={is_sufficient}. Reasoning: {reasoning}"


class Tools: 
    tools = [
        StructuredTool.from_function(name="KnowledgeRetrieverTool", func=knowledge_retriever_tool, description="Fetch domain knowledge"),
        StructuredTool.from_function(name="ArtifactWriterTool", func=artifact_writer_tool, description="Save artifact in plain text to a file with given filename", args_schema=ArtifactWriterInput),
        StructuredTool.from_function(name="ArtifactReaderTool", func=artifact_reader_tool, description="Read artifact from a file with given filename", args_schema=ArtifactReaderInput),
        StructuredTool.from_function(
            func=evaluate_criteria_tool,
            name="EvaluateCriteriaTool",
            description="Evaluate criteria based on the conversation"
        ),
    ]
