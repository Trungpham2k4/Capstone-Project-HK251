import os, json

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import Tool
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field


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

# def knowledge_retriever_tool(query: str) -> str:
#     docs = retriever.get_relevant_documents(query)
#     return "\n".join([d.page_content for d in docs])

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

class Tools: 
    tools = [
        # Tool.from_function(name="KnowledgeRetrieverTool", func=knowledge_retriever_tool, description="Fetch domain knowledge"),
        StructuredTool.from_function(name="ArtifactWriterTool", func=artifact_writer_tool, description="Save artifact in plain text to a file with given filename", args_schema=ArtifactWriterInput),
        StructuredTool.from_function(name="ArtifactReaderTool", func=artifact_reader_tool, description="Read artifact from a file with given filename", args_schema=ArtifactReaderInput)
    ]
