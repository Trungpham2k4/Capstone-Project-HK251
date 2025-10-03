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
import json

from tools import DEPLOYMENT_CRITERIA, artifact_reader_tool

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

deployer_config = {
    "configurable": {
        "thread_id": "deployer_thread"
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
    enduser_done: bool = False
    artifactMetaData: ArtifactMetadata
    interviewer_enduser_messages: Annotated[List[Dict[str, str]], add_messages]

    # Attribute for interview - deployer (temp)
    deployer_phase: int = 0
    deployer_done: bool = False
    interviewer_deployer_messages: Annotated[List[Dict[str, str]], add_messages]
    last_question: str
    last_response: str
    criteria_state: Dict[str, Any]

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
            **state,  # Keep all existing fields
            "artifactMetaData": {"sent_from": "Interviewer", "sent_to": "EndUser"},
            "enduser_phase": new_phase,
            "interviewer_enduser_messages": state["interviewer_enduser_messages"] + [
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


def deployer_interviewer_node(state: ArtifactPoolState, input_text: str, args) -> ArtifactPoolState:
    agent = make_interview_agent(args, dialogue_with="Deployer")

    try:
        url_content = artifact_reader_tool("User_Requirements_List.txt")
    except FileNotFoundError:
        url_content = "User Requirements List not yet available."

    unanswered = [k for k, v in state["criteria_state"].items()
                  if not v["answered"] or not v["sufficient"]]

    if not unanswered:
        print("All criteria completed. Generating Operating Environment List...")

        conversation_summary = "\n\n".join([
            f"{'Interviewer' if i % 2 == 0 else 'Deployer'}: {msg['content']}"
            for i, msg in enumerate(state["interviewer_deployer_messages"])
        ])

        agent.invoke({
            "input": (
                f"=== USER REQUIREMENTS (Context for OEL) ===\n{url_content}\n\n"
                f"=== COMPLETE DEPLOYMENT CONVERSATION ===\n{conversation_summary}\n\n"
                f"=== YOUR TASK ===\n"
                "You have completed interviews covering all 7 deployment criteria:\n"
                "- infrastructure\n"
                "- security\n"
                "- scalability\n"
                "- database\n"
                "- deployment_process\n"
                "- monitoring\n"
                "- compliance\n\n"
                "Now generate a comprehensive Operating Environment List (OEL) by:\n"
                "1. Reviewing the COMPLETE conversation above\n"
                "2. Extracting concrete deployment requirements from each criteria\n"
                "3. For EACH requirement, link it back to relevant user requirements (UR-XXX)\n\n"
                "Call ArtifactWriterTool with:\n"
                "- filename: 'Operating_Environment_List.txt'\n"
                "- data: ALL deployment requirements in the format below\n\n"
                "Required format for EACH requirement:\n"
                "**OE-XXX**: [Short title]\n"
                "* **Category**: [infrastructure/security/scalability/database/deployment_process/monitoring/compliance]\n"
                "* **Description**: [Concrete specification with numbers, tools, versions]\n"
                "* **Justification**: [How this supports UR-XXX from user requirements]\n"
                "* **Source**: Interview with Deployer (Turn [X], criteria: [name])\n\n"
                "Examples:\n"
                "**OE-001**: Multi-Region Cloud Deployment\n"
                "* **Category**: infrastructure\n"
                "* **Description**: Deploy on AWS in us-east-1 and eu-west-1 regions with multi-AZ setup (2 vCPU, 4GB RAM per instance, min 2 replicas per region)\n"
                "* **Justification**: Supports UR-001 (real-time currency conversion) by providing sub-second latency globally\n"
                "* **Source**: Interview with Deployer (Turn 2, criteria: infrastructure)\n\n"
                "**OE-002**: OIDC Authentication with PKCE\n"
                "* **Category**: security\n"
                "* **Description**: Implement OpenID Connect with OAuth 2.0 Authorization Code + PKCE flow, access token lifetime 15 minutes\n"
                "* **Justification**: Supports UR-003 (user trust and security) by providing industry-standard authentication\n"
                "* **Source**: Interview with Deployer (Turn 4, criteria: security)\n\n"
                "Extract ALL requirements from the conversation. Number them sequentially (OE-001, OE-002, ...)."
            ),
            "criteria_list": "",
            "chat_history": state["interviewer_deployer_messages"]
        }, config=deployer_config)

        return {**state, "deployer_done": True}

    criteria_list = "\n".join([
        f"- {k}: {v['description']} [answered: {v['answered']}, sufficient: {v['sufficient']}]"
        for k, v in state["criteria_state"].items()
    ])

    new_phase = state["deployer_phase"] + 1

    response = agent.invoke({
        "input": (
            f"System context: {input_text}\n\n"
            f"=== USER REQUIREMENTS ===\n{url_content}\n\n"
            f"=== DEPLOYMENT CRITERIA CHECKLIST ===\n{criteria_list}\n\n"
            f"Interview phase: {new_phase}\n"
            f"Next criteria to explore: {', '.join(unanswered)}\n\n"
            "Formulate ONE focused question for the Deployer about the next unanswered criteria."
        ),
        "criteria_list": criteria_list,
        "chat_history": state["interviewer_deployer_messages"]
    }, config=deployer_config)

    question = response.get("output")

    return {
        **state,
        "deployer_phase": new_phase,
        "deployer_done": False,
        "interviewer_deployer_messages": state["interviewer_deployer_messages"] + [
            {"role": "assistant", "content": question}
        ],
        "last_question": question,
        "last_response": ""
    }


def evaluation_node(state: ArtifactPoolState, args) -> ArtifactPoolState:
    agent = make_interview_agent(args, dialogue_with="Deployer")

    try:
        url_content = artifact_reader_tool("User_Requirements_List.txt")
    except FileNotFoundError:
        url_content = "User Requirements not available"

    unanswered = [k for k, v in state["criteria_state"].items()
                  if not v["answered"] or not v["sufficient"]]

    if not unanswered:
        return state

    criteria_list_str = "\n".join([
        f"- {k}: {state['criteria_state'][k]['description']}"
        for k in unanswered
    ])

    evaluation_prompt = (
        f"=== USER REQUIREMENTS (Context) ===\n{url_content}\n\n"
        f"=== EVALUATION TASK ===\n"
        f"QUESTION ASKED:\n{state['last_question']}\n\n"
        f"DEPLOYER'S RESPONSE:\n{state['last_response']}\n\n"
        f"UNANSWERED CRITERIA TO EVALUATE:\n{criteria_list_str}\n\n"
        f"INSTRUCTIONS:\n"
        f"1. Review the USER REQUIREMENTS above\n"
        f"2. Analyze the deployer's response to identify which criteria were addressed\n"
        f"3. For EACH criteria that was mentioned (even partially), call EvaluateCriteriaTool separately\n"
        f"4. For each tool call, provide:\n"
        f"   - criteria_key: the exact key from the list above\n"
        f"   - is_answered: true if deployer mentioned this criteria\n"
        f"   - is_sufficient: true if the info is concrete, complete, and adequate for user requirements\n"
        f"   - reasoning: brief explanation linking to user requirements\n\n"
        f"IMPORTANT:\n"
        f"- If the response addresses multiple criteria, call the tool MULTIPLE TIMES\n"
        f"- If a criteria is not mentioned at all, do NOT call the tool for it\n"
        f"- Example: If response mentions 'infrastructure' and 'security', call tool twice\n"
    )

    agent.invoke({
        "input": evaluation_prompt,
        "criteria_list": "",
        "chat_history": []
    }, config=deployer_config)

    # Reload state
    criteria_path = os.path.join(ARTIFACT_DIR, "deployment_criteria_state.json")
    if os.path.exists(criteria_path):
        with open(criteria_path, "r", encoding="utf-8") as f:
            updated = json.load(f)
            state["criteria_state"].update(updated)

    print("\n=== Evaluation Results ===")
    for k, v in state["criteria_state"].items():
        if v["answered"]:
            print(f"{k}: answered={v['answered']}, sufficient={v['sufficient']}")
        else:
            print(f"{k}: not answered yet")

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
        **state,
        "artifactMetaData": {"sent_from": "EndUser", "sent_to": "Interviewer"},
        "interviewer_enduser_messages": [
            {"role": "human", "content": response.get("output")}
        ]
    }


def deployer_node(state: ArtifactPoolState, input_text: str, args) -> ArtifactPoolState:
    """Deployer responding to Interviewer"""
    agent = make_deployer_agent(args)

    if state["interviewer_deployer_messages"]:
        latest_question = state["interviewer_deployer_messages"][-1].content
    else:
        latest_question = "Please describe your deployment environment in detail."

    response = agent.invoke({
        "query": (
            f"System context: {input_text}\n"
            f"Interviewer's question: {latest_question}\n\n"
            "Provide a detailed, technical response about the deployment environment."
        ),
        "chat_history": state["interviewer_deployer_messages"]
    }, config=deployer_config)

    answer = response.get("output")

    return {
        **state,
        "interviewer_deployer_messages": state["interviewer_deployer_messages"] + [
            {"role": "human", "content": answer}
        ],
        "last_response": answer,
        "artifactMetaData": {"sent_from": "Deployer", "sent_to": "Interviewer"}
    }


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

def check_completion(state: ArtifactPoolState) -> Literal["InterviewerDeployer", "END"]:
    all_satisfied = all(
        v["answered"] and v["sufficient"]
        for v in state["criteria_state"].values()
    )
    if all_satisfied or state["deployer_phase"] >= 20:
        return "END"
    return "DeployerInterviewer"

# -------------------
# Build LangGraph workflow
# -------------------
def build_and_run(input_text: str, args):
    graph = StateGraph(ArtifactPoolState)

    graph.add_node("Interviewer", lambda st: interviewer_node(st, input_text, args))
    graph.add_node("EndUser", lambda st: enduser_node(st, input_text, args))

    graph.add_node("DeployerInterviewer", lambda st: deployer_interviewer_node(st, input_text, args))
    graph.add_node("Deployer", lambda st: deployer_node(st, input_text, args))
    graph.add_node("Evaluation", lambda st: evaluation_node(st, args))

    # graph.add_node("Analyst", lambda st: analyst_node(st, args))
    # graph.add_node("Archivist", lambda st: archivist_node(st, args))
    # graph.add_node("Reviewer", lambda st: reviewer_node(st, args))


    # parallel edges
    graph.add_edge(START, "Interviewer")
    # graph.add_edge("Interviewer", "EndUser")

    # Bỏ cmt đoạn này nếu chạy từ đầu
    # graph.add_conditional_edges("Interviewer", interviewer_continue_asking,
    #                             {"EndUser": "EndUser", END: "DeployerInterviewer"})
    # graph.add_edge("EndUser", "Interviewer")

    # Edges for Deployer dialogue
    graph.add_edge("Interviewer", "DeployerInterviewer") # comment dòng này nếu muốn chạy từ đầu
    graph.add_edge("DeployerInterviewer", "Deployer")
    graph.add_edge("Deployer", "Evaluation")
    graph.add_conditional_edges(
        "Evaluation",
        check_completion,
        {"DeployerInterviewer": "DeployerInterviewer", "END": END}
    )
    
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
        "deployer_done": False,
        # thêm criteria_state và messages cho deployer loop
        "criteria_state": DEPLOYMENT_CRITERIA.copy(),
        "interviewer_deployer_messages": [],
        "last_question": "",
        "last_response": ""
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