"""
iredev_langgraph_agentexecutor.py
LangGraph orchestration + LangChain AgentExecutor demo for iReDev workflow.

Workflow:
Input -> Interviewer node
Interviewer -> (parallel and iterative) EndUser + Deployer -> Analyst -> Archivist -> Reviewer -> Final SRS
"""

import os
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv


from agents.interviewer_agent import make_interview_agent
from agents.enduser_agent import make_enduser_agent
from agents.deployer_agent import make_deployer_agent
from agents.analysis_agent import make_analysis_agent
from agents.archivist_agent import make_archivist_agent
from agents.reviewer_agent import make_reviewer_agent


load_dotenv()

# -------------------
# Artifact Pool
# -------------------
ARTIFACT_DIR = "artifact_pool"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

class Artifact(TypedDict):
    id: str
    type: str
    path: str
    version: int
    metadata: Dict[str, Any]

class ArtifactPoolState(TypedDict):
    artifacts: List[Artifact]

# -------------------
# Node functions
# -------------------
def interviewer_node(state: ArtifactPoolState, input_text: str) -> ArtifactPoolState:
    agent = make_interview_agent()
    agent.invoke({
        "input": (
            f"Interview context: {input_text}. "
            "Prepare guiding questions for EndUser and Deployer. "
            "Then call ArtifactWriterTool with JSON like "
            '{"id":"interviewer_guidance","type":"InterviewerGuidance","content":{...}}'
        )
    })
    return state

def enduser_node(state: ArtifactPoolState) -> ArtifactPoolState:

    agent = make_enduser_agent()
    agent.invoke({
        "input": (
            "Produce UserRequirementList in JSON with 3 user-level requirements. "
            "Call ArtifactWriterTool with JSON like "
            '{"id":"user_req_list","type":"UserRequirementList","content":[...]}'
        )
    })
    return state

def deployer_node(state: ArtifactPoolState) -> ArtifactPoolState:

    agent = make_deployer_agent()
    agent.invoke({
        "input": (
            "Produce OperatingEnvironmentList in JSON. "
            "Call ArtifactWriterTool with JSON like "
            '{"id":"op_env_list","type":"OperatingEnvironmentList","content":[...]}'
        )
    })
    return state

def analyst_node(state: ArtifactPoolState) -> ArtifactPoolState:
    agent = make_analysis_agent()
    agent.invoke({
        "input": (
            "Read UserRequirementList and OperatingEnvironmentList using ArtifactReaderTool. "
            "Then call ArtifactWriterTool twice: once with JSON for SystemRequirementList, "
            "and once for RequirementModel."
        )
    })
    return state

def archivist_node(state: ArtifactPoolState) -> ArtifactPoolState:
    agent = make_archivist_agent()
    agent.invoke({
        "input": (
            "Read system_req_list and requirement_model with ArtifactReaderTool. "
            "Generate a Software Requirement Specification (SRS). "
            "Call ArtifactWriterTool with JSON like "
            '{"id":"srs_draft","type":"SRS","content":"..."}'
        )
    })
    return state

def reviewer_node(state: ArtifactPoolState) -> ArtifactPoolState:

    agent = make_reviewer_agent()
    agent.invoke({
        "input": (
            "Read srs_draft using ArtifactReaderTool. "
            "Review and improve it with feedback: Ensure OAuth2 and PCI compliance. "
            "Then call ArtifactWriterTool with JSON like "
            '{"id":"srs_final","type":"FinalSRS","content":"..."}'
        )
    })
    return state

# -------------------
# Build LangGraph workflow
# -------------------
def build_and_run(input_text: str):
    graph = StateGraph(ArtifactPoolState)

    graph.add_node("Interviewer", lambda st: interviewer_node(st, input_text))
    graph.add_node("EndUser", lambda st: enduser_node(st))
    graph.add_node("Deployer", lambda st: deployer_node(st))
    # graph.add_node("Join", lambda st: join_node(st))
    graph.add_node("Analyst", lambda st: analyst_node(st))
    graph.add_node("Archivist", lambda st: archivist_node(st))
    graph.add_node("Reviewer", lambda st: reviewer_node(st))

    # parallel edges
    graph.add_edge(START, "Interviewer")
    graph.add_edge("Interviewer", "EndUser")
    graph.add_edge("EndUser", "Interviewer")
    graph.add_edge("Interviewer", "Deployer")
    graph.add_edge("Deployer", "Interviewer")

    graph.add_edge("EndUser", "Analyst")
    graph.add_edge("Deployer", "Analyst")

    # linear flow
    graph.add_edge("Analyst", "Archivist")
    graph.add_edge("Archivist", "Reviewer")
    graph.add_edge("Reviewer", END)

    app = graph.compile()
    init_state: ArtifactPoolState = {"artifacts":[]}
    final_state = app.invoke(init_state)
    return final_state

# -------------------
# Run example
# -------------------
if __name__ == "__main__":
    input_text = "Build an online bookstore with secure payments and good search UX."
    final_state = build_and_run(input_text)

    print("\n=== Final artifacts in pool ===")
    for fname in os.listdir(ARTIFACT_DIR):
        print("-", fname)


### TODO: Sửa lại phần update state của ArtifactPoolState trong mỗi node function để thêm artifact mới vào list artifacts.
### TODO: Thêm hội thoại đa lượt giữa Interviewer <-> EndUser và Interviewer <-> Deployer.