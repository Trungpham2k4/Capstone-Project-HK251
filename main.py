"""
LangGraph orchestration + LangChain AgentExecutor demo for iReDev workflow.

Workflow:
Input -> Interviewer node 
Interviewer <-> EndUser => Conditional edge to continue asking or go to ask Deployer -> Create UserRequirementList
Interviewer <-> Deployer => Conditional edge to continue asking or go to Analyst -> Create OperatingEnvironmentList
UserRequirementList + OperatingEnvironmentList -> Analyst -> Create SystemRequirementList + RequirementModel
SystemRequirementList + RequirementModel -> Archivist -> Create SRS draft
SRS draft -> Reviewer -> Final SRS
"""

import os
import argparse
from typing import TypedDict, List, Dict, Any, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
load_dotenv()

from agents.interviewer_agent import make_interview_agent
from agents.enduser_agent import make_enduser_agent
from agents.deployer_agent import make_deployer_agent
from agents.analysis_agent import make_analysis_agent
from agents.archivist_agent import make_archivist_agent
from agents.reviewer_agent import make_reviewer_agent




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
    # messages: Annotated[list, add_messages]

# -------------------
# Node functions
# -------------------
def interviewer_node(state: ArtifactPoolState, input_text: str, args) -> ArtifactPoolState:
    agent = make_interview_agent(args)
    agent.invoke({
        "input": (
            f"Interview context: {input_text}. "
            "Prepare guiding questions for EndUser and Deployer. "
            "Then call ArtifactWriterTool with JSON like "
            '{"id":"interviewer_guidance","type":"InterviewerGuidance","content":{...}}'
        )
    })
    return state

def enduser_node(state: ArtifactPoolState, args) -> ArtifactPoolState:

    agent = make_enduser_agent(args)
    agent.invoke({
        "input": (
            "Produce UserRequirementList in JSON with 3 user-level requirements. "
            "Call ArtifactWriterTool with JSON like "
            '{"id":"user_req_list","type":"UserRequirementList","content":[...]}'
        )
    })
    return state

def deployer_node(state: ArtifactPoolState, args) -> ArtifactPoolState:

    agent = make_deployer_agent(args)
    agent.invoke({
        "input": (
            "Produce OperatingEnvironmentList in JSON. "
            "Call ArtifactWriterTool with JSON like "
            '{"id":"op_env_list","type":"OperatingEnvironmentList","content":[...]}'
        )
    })
    return state

def analyst_node(state: ArtifactPoolState, args) -> ArtifactPoolState:
    agent = make_analysis_agent(args)

    messages = {
        "input" : (
            "Please see the latest User Requirements List and Operating Environment List, and produce a consolidated System Requirements List and Requirements Model as your system message."
        )
    }

    response = agent.invoke(messages)
    return {"messages": [{"role": "assistant", "content": response.get("output")}]}

def archivist_node(state: ArtifactPoolState, args) -> ArtifactPoolState:

    messages = { 
        "input" : (
            "Please see the latest System Requirements List and Requirements Model, and draft a structured Software Requirements Specification (SRS) as your system message."
        ) 
    }

    agent = make_archivist_agent(args)
    response = agent.invoke(messages)
    return {"messages": [{"role": "assistant", "content": response.get("output")}]}

def reviewer_node(state: ArtifactPoolState, arg) -> ArtifactPoolState:

    messages = { 
        "input" : (
            "Please review the latest SRS draft, fix any issues and output the revised software requirements specification."
        ) 
    }

    agent = make_reviewer_agent(arg)
    response = agent.invoke(messages)
    return {"messages": [{"role": "assistant", "content": response.get("output")}]}

# -------------------
# Build LangGraph workflow
# -------------------
def build_and_run(input_text: str, args):
    graph = StateGraph(ArtifactPoolState)

    # graph.add_node("Interviewer", lambda st: interviewer_node(st, input_text))
    # graph.add_node("EndUser", lambda st: enduser_node(st))
    # graph.add_node("Deployer", lambda st: deployer_node(st))
    graph.add_node("Analyst", lambda st: analyst_node(st, args))
    graph.add_node("Archivist", lambda st: archivist_node(st, args))
    graph.add_node("Reviewer", lambda st: reviewer_node(st, args))

    # parallel edges
    # graph.add_edge(START, "Interviewer")
    # graph.add_edge("Interviewer", "EndUser")
    # graph.add_edge("EndUser", "Interviewer")
    # graph.add_edge("Interviewer", "Deployer")
    # graph.add_edge("Deployer", "Interviewer")

    # graph.add_edge("EndUser", "Analyst")
    # graph.add_edge("Deployer", "Analyst")

    # linear flow
    graph.add_edge(START, "Analyst")
    graph.add_edge("Analyst", "Archivist")
    graph.add_edge("Archivist", "Reviewer")
    graph.add_edge("Reviewer", END)

    app = graph.compile()
    # init_state: ArtifactPoolState = {"artifacts":[]}
    init_state: ArtifactPoolState = {"messages":[]}
    final_state = app.invoke(init_state)
    return final_state

# -------------------
# Run example
# -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', default="gpt-5-nano", type=str, help="The model name")
    parser.add_argument('--model_base_url', default=None, type=str, help="The model base URL")
    parser.add_argument('--model_temperature', default=0, type=float, help="The model temperature")

    args = parser.parse_args()

    input_text = "Build an online bookstore with secure payments and good search UX."
    final_state = build_and_run(input_text, args)


### TODO: Sửa lại phần update state của ArtifactPoolState trong mỗi node function để thêm artifact mới vào list artifacts.
### TODO: Thêm hội thoại đa lượt giữa Interviewer <-> EndUser và Interviewer <-> Deployer.