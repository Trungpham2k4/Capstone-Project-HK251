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
from typing import Literal, TypedDict, List, Dict, Any, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
load_dotenv()

from agents.interviewer_agent import make_interview_agent
from agents.enduser_agent import make_enduser_agent
from agents.deployer_agent import make_deployer_agent
from agents.analysis_agent import make_analysis_agent
from agents.archivist_agent import make_archivist_agent
from agents.reviewer_agent import make_reviewer_agent

# -------------------
# Memory
# -------------------
memory = MemorySaver()
enduser_config = {
    "configurable" : {
        "thread_id": "1"
    }
}

# -------------------
# Artifact Pool
# -------------------
ARTIFACT_DIR = "artifact_pool"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

class ArtifactMetadata(TypedDict):
    sent_from: Literal["Human", "Interviewer", "EndUser", "Deployer", "Analyst", "Archivist"]
    sent_to: Literal["Interviewer", "EndUser", "Deployer", "Analyst", "Archivist", "Reviewer"]

class ArtifactPoolState(TypedDict):
    enduser_phase: int = 0
    deployer_phase: int = 0
    enduser_done: bool = False
    deployer_done: bool = False
    artifactMetaData: ArtifactMetadata
    interviewer_enduser_messages: Annotated[List[Dict[str, str]], add_messages]

# -------------------
# Node functions
# -------------------
def interviewer_node(state: ArtifactPoolState, input_text: str, args) -> ArtifactPoolState:
    agent = None
    if state["artifactMetaData"]["sent_from"] == "Human" or state["artifactMetaData"]["sent_from"] == "EndUser":
        agent = make_interview_agent(args, dialogue_with="EndUser")

    if state["enduser_phase"] < 15:
        new_phase = state["enduser_phase"] + 1

        response = agent.invoke({
            "input": (
                f"Interview context: {input_text}. "
                f"Conversation phase (turn): {new_phase}.\n"
                "Prepare a dedicated question for EndUser based on the previous conversation and your system message."
            ),
            "chat_history": state["interviewer_enduser_messages"]
        }, config=enduser_config)

        # print("Interviewer response:", response)

        return {
            "artifactMetaData": {"sent_from": "Interviewer", "sent_to": "EndUser"},
            "enduser_phase": new_phase,   # <-- giữ phase mới
            "enduser_done": state["enduser_done"],  # <-- giữ lại flag cũ
            "deployer_phase": state["deployer_phase"], 
            "deployer_done": state["deployer_done"],
            "interviewer_enduser_messages": [
                {"role": "assistant", "content": response.get("output")}
            ]
        }

    else:
        agent.invoke({
            "input": (
                f"Interview context: {input_text}. "
                f"Conversation phase (turn): {state['enduser_phase']}.\n"
                f"Generate User Requirements List based on conversation with EndUser"
            ),
            "chat_history": state["interviewer_enduser_messages"]
        }, config=enduser_config)
        return state

    # elif state["deployer_phase"] < 3 and state["enduser_phase"] >= 3:
    #     response = agent.invoke({
    #         "input": (
    #             f"Interview context: {input_text}. "
    #             "Prepare guiding questions for Deployer "
    #         )
    #     })
    #     state["artifactMetaData"]["sent_from"] = "Interviewer"
    #     state["artifactMetaData"]["sent_to"] = "Deployer"
    # elif state["enduser_phase"] >= 3 and state["deployer_phase"] >= 3:
    #     state["artifactMetaData"]["sent_from"] = "Interviewer"
    #     state["artifactMetaData"]["sent_to"] = "Analyst"
    return state

def enduser_node(state: ArtifactPoolState, input_text: str, args) -> ArtifactPoolState:
    agent = make_enduser_agent(args)
    lastest_question = state["interviewer_enduser_messages"][-1].content if state["interviewer_enduser_messages"] else ""
    response = agent.invoke({
        "input": (
            f"Interview context: {input_text}. "
            f"Respond to the interviewer's question: {lastest_question} based on your persona and scenario context."
        ),
        "chat_history": state["interviewer_enduser_messages"]
    }, config=enduser_config)
    # print("EndUser response:", response)

    return {
        "artifactMetaData": {"sent_from": "EndUser", "sent_to": "Interviewer"},
        "enduser_phase": state["enduser_phase"],
        "interviewer_enduser_messages": [
            {"role": "human", "content": response.get("output")}
        ]
    }


def deployer_node(state: ArtifactPoolState, args) -> ArtifactPoolState:

    agent = make_deployer_agent(args)
    agent.invoke({
        "input": (

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

def interviewer_continue_asking(state: ArtifactPoolState) -> Literal["EndUser"]:
    if state["artifactMetaData"]["sent_to"] == "EndUser" and state["enduser_phase"] <= 15:
        return "EndUser"
    return END

# -------------------
# Build LangGraph workflow
# -------------------
def build_and_run(input_text: str, args):
    graph = StateGraph(ArtifactPoolState)

    graph.add_node("Interviewer", lambda st: interviewer_node(st, input_text, args))
    graph.add_node("EndUser", lambda st: enduser_node(st, input_text, args))
    # graph.add_node("Deployer", lambda st: deployer_node(st, args))
    # graph.add_node("Analyst", lambda st: analyst_node(st, args))
    # graph.add_node("Archivist", lambda st: archivist_node(st, args))
    # graph.add_node("Reviewer", lambda st: reviewer_node(st, args))

    # parallel edges
    graph.add_edge(START, "Interviewer")
    # graph.add_edge("Interviewer", "EndUser")
    graph.add_conditional_edges("Interviewer", interviewer_continue_asking, {"EndUser": "EndUser", END: "__end__"})
    graph.add_edge("EndUser", "Interviewer")
    
    # graph.add_edge("Interviewer", "Deployer")
    # graph.add_edge("Deployer", "Interviewer")

    # graph.add_edge("EndUser", "Analyst")
    # graph.add_edge("Deployer", "Analyst")

    # linear flow
    # graph.add_edge(START, "Analyst")
    # graph.add_edge("Analyst", "Archivist")
    # graph.add_edge("Archivist", "Reviewer")
    # graph.add_edge("Reviewer", END)

    app = graph.compile(checkpointer=memory)
    init_state: ArtifactPoolState = {
        "artifactMetaData": {
            "sent_from": "Human", 
            "sent_to": "Interviewer"
        }, 
        "interviewer_enduser_messages": [], 
        "enduser_phase": 0, 
        "deployer_phase": 0,
        "enduser_done": False,
        "deployer_done": False
    }
    final_state = app.invoke(init_state, {"configurable": {"thread_id": "1"}, "recursion_limit": 40})
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

    # input_text = "Build an online bookstore with secure payments and good search UX."
    input_text = "I need a currency converter webpage."
    final_state = build_and_run(input_text, args)
    print(final_state)


### TODO: Thêm hội thoại đa lượt giữa Interviewer <-> EndUser và Interviewer <-> Deployer.