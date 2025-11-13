# -------------------------
# Thinking module
# -------------------------

from typing import Dict, Any, Optional

from agents.analyst_agent.profile import AnalystProfile
from agents.analyst_agent.knowledge import AnalystKnowledge
from agents.analyst_agent.memory import AnalystMemory
from agents.analyst_agent.action import AnalystAction

from agents.base_agent.thinking import ThinkingModule
from openai import OpenAI

ALLOWED_ACTIONS_ANALYST = {"generate_system_requirements", "choose_requirement_model", "generate_requirement_model"}

class AnalystThinking(ThinkingModule):

    def __init__(self, profile: AnalystProfile, knowledge: AnalystKnowledge,
                 memory: AnalystMemory, action: AnalystAction, llm_client: OpenAI):
        self.profile = profile
        self.knowledge = knowledge
        self.memory = memory
        self.action = action
        self.llm = llm_client

    def decide(self, message: Dict[str, Any]):

        print(f"\n[Thinking] Starting decision process for message from {message.get('sent_from')}")
        
        # Decision-Action loop
        while True:

            # 1. Make decision
            decision = self._make_decision(message)
            
            if not decision:
                print("[Thinking] Failed to make valid decision, stopping.")
                break
            
            # 2. Execute action
            execution_result = self.action.execute(decision, message=message)
            
            # 4. Check execution status
            status = execution_result.get("status")
            
            if status == "complete":
                # print("[Thinking] Process completed successfully")
                break
            elif status == "continue":
                print("[Thinking] Continuing decision process...")
                continue
            elif status == "error":
                print(f"[Thinking] Error occurred: {execution_result.get('reason')}")
                break

    def _make_decision(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Make a single decision based on current message state.
        """
        prompt = self._build_analyst_prompt(message)
        allowed_actions = ALLOWED_ACTIONS_ANALYST

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

    def _build_analyst_prompt(self, message: Dict[str, Any]) -> str:
        """Build prompt for Analyst agent decision-making."""
        print("[Thinking] Building analyst prompt...")

        # Build status indicators
        system_requirement_content, system_requirement_generated = self.memory.read("system_requirements")
        requirement_model_content, requirement_model_chosen = self.memory.read("requirement_model")

        system_requirement_status = "✓ GENERATED" if system_requirement_generated else "✗ NOT GENERATED"
        requirement_model_status = f"✓ CHOSEN" if requirement_model_chosen else "✗ NOT CHOSEN"


        # Knowledge (simplified)
        kb_text = "No relevant knowledge found."

        prompt = f"""

        CURRENT STATE:
        - System Requirements Status: {system_requirement_status}
        - Requirement Model Chosen Status: {requirement_model_status}

        CHOSEN REQUIREMENT MODEL: 
        {requirement_model_content if requirement_model_chosen else "N/A"}

        CONTEXT:
        - Knowledge context: {kb_text}

        ALLOWED ACTIONS (choose EXACTLY ONE):
        - generate_system_requirements: create a consistent system requirements list from user requirement list and operating environment list.
        - choose_requirement_model: select an appropriate requirement modeling methodology (e.g., UML, SysML-v2).
        - generate_requirement_model: create a requirement model using textual syntax diagram (PlantUML, SysML-v2) based on chosen modeling methodology and system requirements list.

        MANDATORY DECISION LOGIC - FOLLOW EXACTLY:

            IF System Requirements NOT GENERATED:
                → MUST choose: generate_system_requirements
            ELSE IF System Requirements GENERATED AND Requirement Model NOT CHOSEN:
                → MUST choose: choose_requirement_model
            ELSE IF System Requirements GENERATED AND Requirement Model CHOSEN:
                → MUST choose: generate_requirement_model

        OUTPUT FORMAT (strict JSON only):
        {{
            "rationale": "Step-by-step instructions on how to do the chosen action (2-3 paragraphs)",
            "action": "<one of the allowed actions>"
        }}
        """
        return prompt