# -------------------------
# Thinking module
# -------------------------

from typing import Dict, Any, Optional

from agents.archivist_agent.profile import ArchivistProfile
from agents.archivist_agent.knowledge import ArchivistKnowledge
from agents.archivist_agent.memory import ArchivistMemory
from agents.archivist_agent.action import ArchivistAction

from agents.base_agent.thinking import ThinkingModule
from openai import OpenAI

ALLOWED_ACTIONS_ARCHIVIST = {"generate_software_requirements_specification"}

class ArchivistThinking(ThinkingModule):

    def __init__(self, profile: ArchivistProfile, knowledge: ArchivistKnowledge,
                 memory: ArchivistMemory, action: ArchivistAction, llm_client: OpenAI):
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
            
            # 3. Check execution status
            status = execution_result.get("status")
            
            if status == "complete":
                # print("[Thinking] Process completed successfully")
                break
            elif status == "error":
                print(f"[Thinking] Error occurred: {execution_result.get('reason')}")
                break

    def _make_decision(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Make a single decision based on current message state.
        """
        prompt = self._build_archivist_prompt(message)
        allowed_actions = ALLOWED_ACTIONS_ARCHIVIST

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

    def _build_archivist_prompt(self, message: Dict[str, Any]) -> str:
        """Build prompt for Archivist agent decision-making."""
        print("[Thinking] Building archivist prompt...")


        # Knowledge (simplified)
        kb_text = "No relevant knowledge found."

        prompt = f"""

        CONTEXT:
        - Knowledge context: {kb_text}

        ALLOWED ACTIONS (choose EXACTLY ONE):
        - generate_software_requirements_specification: Generate a comprehensive software requirements specification document based on system requirements list and requirements model

        OUTPUT FORMAT (strict JSON only):
        {{
            "rationale": "Step-by-step instructions on how to do the chosen action (2-3 paragraphs)",
            "action": "<one of the allowed actions>"
        }}
        """
        return prompt