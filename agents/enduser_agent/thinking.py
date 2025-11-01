# -------------------------
# Thinking module
# -------------------------

from typing import Dict, Any, Optional

from agents.enduser_agent.profile import EndUserProfile
from agents.enduser_agent.knowledge import EndUserKnowledge
from agents.enduser_agent.memory import EndUserMemory
from agents.enduser_agent.action import EndUserAction

from agents.base_agent.thinking import ThinkingModule
from openai import OpenAI

### Idea for interaction between ThinkingModule and ActionModule:
### Build prompt in Thinking module to get next action and reasoning process from LLM 
### Action module executes the action with reasoning provided by Thinking module, after finishing the action,
### It can send back to Thinking module and continue to reason about next steps or finish.

ALLOWED_ACTIONS_ENDUSER = {"respond", "clarify"}

class EndUserThinking(ThinkingModule):
    """
    The Thinking module integrates profile, knowledge, and memory to guide reasoning.
    It determines ONE next action and provides rationale (reasoning chain).
    "action": string
    "reasoning": string
    """

    def __init__(self, profile: EndUserProfile, knowledge: EndUserKnowledge,
                 memory: EndUserMemory, action: EndUserAction, llm_client: OpenAI):
        self.profile = profile
        self.knowledge = knowledge
        self.memory = memory
        self.action = action
        self.llm = llm_client
        self.conversation_turns = 1
        self.user_input = "" # Cái này chưa có tí phải thêm bằng cách nào đó

    def decide(self, artifact: Dict[str, Any]):
        """
        Main decision loop: Think → Act → Check status → Repeat if needed.
        Tracks conversation turns: increments only when ask_question is executed.
        """
        print(f"\n[Thinking] Starting decision process for artifact from {artifact.get('sent_from')}")
        print(f"[Thinking] Current conversation turns: {self.conversation_turns}")
        
        # Initial decision
        current_artifact = artifact.copy()
        
        # Decision-Action loop
        while True:

            if self.conversation_turns > 10:
                print("[Thinking] Maximum conversation turns reached, generate artifacts.")
                self.action.execute({"action" : "generate_user_requirements", "rationale": "Max conversation turns exceeded"}, current_artifact)
                self.conversation_turns = 1
                break

            # 1. Make decision
            decision = self._make_decision(current_artifact)
            
            if not decision:
                print("[Thinking] Failed to make valid decision, stopping.")
                break
            
            # 2. Execute action
            execution_result = self.action.execute(decision, current_artifact)
            
            # 3. Update conversation turns if ask_question was executed
            if decision.get("action") == "ask_question" and execution_result.get("status") in ["waiting", "complete"]:
                self.conversation_turns += 1
                self.retrieve_record_done = False  # Reset for next turn
                self.saturation_evaluated = False  # Reset for next turn
                self.saturation_score = None
                self.saturation_reasoning = ""
                self.record_text = ""
                # print(f"[Thinking] Conversation turn incremented to: {self.conversation_turns}")

            if decision.get("action") == "generate_user_requirements":
                self.conversation_turns = 1  # Reset after generating requirements
            
            # 4. Check execution status
            status = execution_result.get("status")
            
            # print(f"[Thinking] Action result status: {status}")
            
            if status == "complete":
                # print("[Thinking] Process completed successfully")
                break
            elif status == "error":
                print(f"[Thinking] Error occurred: {execution_result.get('reason')}")
                break
            elif status == "waiting":
                # Action requires external input (e.g., waiting for user response)
                # print("[Thinking] Waiting for external input, pausing decision loop")
                break
            elif status == "continue":
                # Action completed, but process should continue with next decision
                print("[Thinking] Continuing to next decision...")
                
                # Update artifact with execution result if needed
                if "data" in execution_result:
                    current_artifact["context_data"] = execution_result["data"]
                    if execution_result["action"] == "retrieve_interview_record":
                        self.retrieve_record_done = True
                        self.record_text = execution_result["data"].get("record_text", "")
                    if execution_result["action"] == "evaluate_saturation":
                        self.saturation_evaluated = True
                        self.saturation_score = execution_result["data"].get("saturation_score", None)
                        self.saturation_reasoning = execution_result["data"].get("reasoning", "")
            else:
                print(f"[Thinking] Unknown status: {status}, stopping")
                break
        
        # print(f"[Thinking] Decision process finished.")
        # print(f"[Thinking] Total conversation turns: {self.conversation_turns}\n")

    def _make_decision(self, artifact: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Make a single decision based on current artifact state.
        """
        prompt = self._build_enduser_prompt(artifact)
        allowed_actions = ALLOWED_ACTIONS_ENDUSER

        # Get decision from LLM
        try:
            response = self.llm.responses.create(
                model="gpt-5-nano",
                input=[
                    {"role": "system", "content": self.profile.system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                store=True,
                reasoning={"effort": "medium"},
                text={
                    "format": {
                        "type": "json_schema",
                        "strict": True,
                        "name": "DecisionOutput",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "rationale": {"type": "string"},
                                "action": {"type": "string"}
                            },
                            "required": ["rationale", "action"],
                            "additionalProperties": False
                        }
                    }
                }
            )

            raw_output = response.output_text
            print(f"[Thinking] LLM raw output: {raw_output[:200]}...")
            
        except Exception as e:
            print(f"[Thinking] Error calling LLM: {e}")
            return None
        
        # Parse and validate decision
        decision = self.parse_and_validate_decision(raw_output, allowed_actions)
        
        if not decision:
            print("[Thinking] Invalid decision from LLM, using default")
            decision = {
                "rationale": "Default action: provide response",
                "action": "respond"
            }
        
        return decision

    def _build_enduser_prompt(self, artifact: Dict[str, Any]) -> str:
        """Build prompt for EndUser agent decision-making."""
        print("[Thinking] Building enduser prompt...")
        
        question = artifact.get("content", "")
        
        # Knowledge and memory (simplified)
        kb_text = "No relevant knowledge found."
        mem_text = "No recent memory."
        
        if self.knowledge:
            try:
                kb_snips = self.knowledge.retrieve(question, k=3)
                if kb_snips:
                    kb_text = "\n".join(f"- {s.get('text', '')}" for s in kb_snips)
            except:
                pass
        
        if self.memory:
            try:
                mem_snips = self.memory.semantic_search(question, top_k=3)
                if mem_snips:
                    mem_text = "\n".join(f"- {m.get('content', '')}" for m in mem_snips)
            except:
                pass

        prompt = f"""

        QUESTION FROM INTERVIEWER:
        "{question}"

        GOAL:
        - Answer the interviewer's question clearly and provide relevant details about your needs.

        CONTEXT:
        - Artifact type: {artifact.get('type')}
        - From: {artifact.get('sent_from')}
        - Knowledge context: {kb_text}
        - Memory context: {mem_text}
        - Main context: "{self.user_input}"

        ALLOWED ACTIONS (choose EXACTLY ONE):
        - respond: provide the answer text and recipients.
        required fields: type, content, recipients, artifact_type
        - clarify: ask interviewer for clarification (if question ambiguous).

        DECISION RULES:
        - If question is clear: respond with relevant details
        - If question is ambiguous: ask for clarify
        - If question is outside your knowledge: respond with what you know
        - Always try to provide examples and specific details


        OUTPUT FORMAT (strict JSON only):
        {{
            "rationale": "<short reasoning steps, 2–5 sentences>",
            "action": "<one of the allowed actions>"
        }}
        """
        return prompt
