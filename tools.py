import os, json

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import Tool


ARTIFACT_DIR = "artifact_pool"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def save_artifact(artifact_id: str, content: dict, artifact_type: str):
    path = os.path.join(ARTIFACT_DIR, f"{artifact_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=2)
    art = {"id": artifact_id, "type": artifact_type,
           "path": path, "version": 1, "metadata": {}}
    return art

def load_artifact(artifact_id: str) -> dict:
    path = os.path.join(ARTIFACT_DIR, f"{artifact_id}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# -------------------
# Build retriever + tools
# -------------------
llm = ChatOpenAI(model="gpt-5-nano", temperature=0, max_tokens=500)
emb = OpenAIEmbeddings(model="text-embedding-3-small")
vs = FAISS.from_texts([
    "Payment must follow PCI DSS; workflow: auth -> capture -> settle.",
    "Deployment env: 2 web servers, 1 DB cluster, TLS, port 443.",
    "Security: must support OAuth2, RBAC, password min length 8."
], emb)
retriever = vs.as_retriever(search_kwargs={"k": 2})

def knowledge_retriever_tool(query: str) -> str:
    docs = retriever.get_relevant_documents(query)
    return "\n".join([d.page_content for d in docs])

def artifact_writer_tool(data: str) -> str:
    try:
        obj = json.loads(data)
    except Exception:
        obj = {"raw": data}
    artifact_id = obj.get("id", "artifact_" + str(len(os.listdir(ARTIFACT_DIR))+1))
    art = save_artifact(artifact_id, obj, obj.get("type","Unknown"))
    return f"Saved {artifact_id} ({art['type']})"

def artifact_reader_tool(artifact_id: str) -> str:
    return json.dumps(load_artifact(artifact_id), ensure_ascii=False)


class Tools: 
    tools = [
        Tool.from_function(name="KnowledgeRetrieverTool", func=knowledge_retriever_tool, description="Fetch domain knowledge"),
        Tool.from_function(name="ArtifactWriterTool", func=artifact_writer_tool, description="Save artifact JSON"),
        Tool.from_function(name="ArtifactReaderTool", func=artifact_reader_tool, description="Read artifact JSON by ID"),
    ]
