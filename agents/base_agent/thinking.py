# -------------------------
# Thinking module
# -------------------------

from typing import Dict, Any, Optional

from agents.base_agent.profile import ProfileModule
from agents.base_agent.knowledge import KnowledgeModule
from agents.base_agent.memory import MemoryModule
from agents.base_agent.action import ActionModule
from openai import OpenAI

### Idea for interaction between ThinkingModule and ActionModule:
### Build prompt in Thinking module to get next action and reasoning process from LLM 
### Action module executes the action with reasoning provided by Thinking module, after finishing the action,
### It can send back to Thinking module and continue to reason about next steps or finish.


import json, re

class ThinkingModule:
    """
    The Thinking module integrates profile, knowledge, and memory to guide reasoning.
    It determines ONE next action and provides rationale (reasoning chain).
    "action": string
    "reasoning": string
    """

    def __init__(self, profile: ProfileModule, knowledge: KnowledgeModule,
                 memory: MemoryModule, action: ActionModule, llm_client: OpenAI):
        self.profile = profile
        self.knowledge = knowledge
        self.memory = memory
        self.action = action
        self.llm = llm_client

    def decide(self, message: Dict[str, Any]):
        """
        Main decision loop: Think → Act → Check status → Repeat if needed.
        Tracks conversation turns: increments only when ask_question is executed.
        """
        pass

    def _make_decision(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Make a single decision based on current message state.
        """
        pass

    @staticmethod
    def parse_and_validate_decision(raw_text: str, allowed_actions: set) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM output and validate structure."""
        if not raw_text:
            return None
        
        # Try to extract JSON
        try:
            # Direct parse
            data = json.loads(raw_text)
        except:
            # Try to find JSON in markdown code blocks
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                except:
                    return None
            else:
                # Try to find any JSON object
                match = re.search(r'\{.*\}', raw_text, re.DOTALL)
                if not match:
                    return None
                try:
                    data = json.loads(match.group(0))
                except:
                    return None
        
        if not isinstance(data, dict):
            return None
        
        action = data.get("action", "").strip()
        rationale = data.get("rationale", "").strip()
        
        # Validate action
        if not action or action not in allowed_actions:
            print(f"[Thinking] Action '{action}' not in allowed set {allowed_actions}")
            return None
        
        return {
            "rationale": rationale,
            "action": action,
            "metadata": data.get("metadata", {})
        }


