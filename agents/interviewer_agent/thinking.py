# -------------------------
# Thinking module
# -------------------------

from typing import Dict, Any, Optional

from agents.interviewer_agent.profile import InterviewerProfile
from agents.interviewer_agent.knowledge import InterviewerKnowledge
from agents.interviewer_agent.memory import InterviewerMemory
from agents.interviewer_agent.action import InterviewerAction

from agents.base_agent.thinking import ThinkingModule
from openai import OpenAI

### Idea for interaction between ThinkingModule and ActionModule:
### Build prompt in Thinking module to get next action and reasoning process from LLM 
### Action module executes the action with reasoning provided by Thinking module, after finishing the action,
### It can send back to Thinking module and continue to reason about next steps or finish.

ALLOWED_ACTIONS_INTERVIEWER = {"ask_question","generate_user_requirements","evaluate_saturation","retrieve_interview_record"}

class InterviewerThinking(ThinkingModule):
    """
    The Thinking module integrates profile, knowledge, and memory to guide reasoning.
    It determines ONE next action and provides rationale (reasoning chain).
    "action": string
    "reasoning": string
    """

    def __init__(self, profile: InterviewerProfile, knowledge: InterviewerKnowledge,
                 memory: InterviewerMemory, action: InterviewerAction, llm_client: OpenAI):
        self.profile = profile
        self.knowledge = knowledge
        self.memory = memory
        self.action = action
        self.llm = llm_client
        self.conversation_turns = 1
        self.user_input = ""

        self.saturation_evaluated = False
        self.retrieve_record_done = False
        self.record_text = ""
        self.saturation_score = None
        self.saturation_reasoning = ""

    def decide(self, message: Dict[str, Any]):
        """
        Main decision loop: Think → Act → Check status → Repeat if needed.
        Tracks conversation turns: increments only when ask_question is executed.
        """
        print(f"\n[Thinking] Starting decision process for message from {message.get('sent_from')}")
        print(f"[Thinking] Current conversation turns: {self.conversation_turns}")
        
        # Decision-Action loop
        while True:

            if self.conversation_turns > 10:
                print("[Thinking] Maximum conversation turns reached, generate messages.")
                self.action.execute({"action" : "generate_user_requirements", "rationale": "Max conversation turns exceeded"}, message)
                self.conversation_turns = 1
                break

            # 1. Make decision
            decision = self._make_decision(message=message)
            
            if not decision:
                print("[Thinking] Failed to make valid decision, stopping.")
                break
            
            # 2. Execute action
            execution_result = self.action.execute(decision, message)
            
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
                
                # Update message with execution result if needed
                if "data" in execution_result:
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

    def _make_decision(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Make a single decision based on current message state.
        """
        # Determine which prompt to use based on roles
        sent_from = message.get("sent_from")
        sent_to = message.get("sent_to")
        
        if sent_from == "User":
            self.user_input = message.get("content", "")

        if (sent_from == "Enduser" and sent_to == "Interviewer") or (sent_from == "User" and sent_to == "Interviewer"):
            prompt = self._build_interviewer_prompt(message)
            allowed_actions = ALLOWED_ACTIONS_INTERVIEWER
        else:
            print(f"[Thinking] Unknown role direction: {sent_from} → {sent_to}")
            return None
        
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
            # Default fallback based on role
            if sent_from == "Enduser":
                decision = {
                    "rationale": "Default action: ask follow-up question",
                    "action": "ask_question"
                }
        
        return decision
    
    def _build_interviewer_prompt(self, message: Dict[str, Any]) -> str:
        """Build prompt for Interviewer agent decision-making."""

        print("[Thinking] Building interviewer prompt...")
        
        # Get context data if available (from previous actions like retrieve_interview_record)
        # context_data = message.get("context_data", {})
        conversation_turns = self.conversation_turns
        
        # Retrieve relevant knowledge and memory (simplified for now)
        content = message.get("content", "")


        # Interview record data (if retrieved)
        # record_text = context_data.get("record_text", "")
        
        # Saturation data (if evaluated)
        # saturation_score = context_data.get("saturation_score", None)
        # saturation_reasoning = context_data.get("reasoning", "")

        

        # Build context sections
        record_section = ""
        if self.record_text:
            record_section = f"""
                INTERVIEW RECORD:
                {self.record_text}
            """
        
        saturation_section = ""
        if self.saturation_score is not None:
            saturation_section = f"""
                SATURATION EVALUATION:
                - Score: {self.saturation_score:.2f}
                - Reasoning: {self.saturation_reasoning}
                """

        # Build status indicators
        record_status = "✓ RETRIEVED" if self.retrieve_record_done else "✗ NOT RETRIEVED"
        saturation_status = f"✓ EVALUATED (score: {self.saturation_score:.2f})" if self.saturation_evaluated else "✗ NOT EVALUATED"

        print(f"[Thinking] Record status: {record_status}, Saturation status: {saturation_status}")

        # Knowledge retrieval (placeholder - replace with actual implementation)
        kb_text = "No relevant prior knowledge found."
        if self.knowledge:
            try:
                kb_snips = self.knowledge.retrieve(content, k=3)
                if kb_snips:
                    kb_text = "\n".join(f"- {s.get('text', '')}" for s in kb_snips)
            except:
                pass
        
        # Memory retrieval (placeholder)
        mem_text = "No recent memory retrieved."
        if self.memory:
            try:
                mem_snips = self.memory.semantic_search(content, top_k=3)
                if mem_snips:
                    mem_text = "\n".join(f"- {m.get('content', '')}" for m in mem_snips)
            except:
                pass

        prompt = f"""You are an Interviewer Agent conducting requirements elicitation.

            CURRENT STATE:
            - Conversation Turn: {conversation_turns}
            - Last Enduser Response: "{content}"
            - Interview Record Status: {record_status}
            - Saturation Evaluation Status: {saturation_status}

            {record_section}

            {saturation_section}

            CONTEXT:
            - Knowledge context: {kb_text}
            - Memory context: {mem_text}
            - Main context input: "{self.user_input}"

            ALLOWED ACTIONS (choose EXACTLY ONE):
            - ask_question: ask a clarifying or exploratory question to the stakeholder.
            - generate_user_requirements: create a draft User Requirements List from conversation.
            - evaluate_saturation: analyze conversation for saturation; return a short score & recommendation.
            - retrieve_interview_record: request the tool to read conversation (ActionModule will call retrieve_interview_record).

            MANDATORY DECISION LOGIC - FOLLOW EXACTLY:

            IF conversation_turns == 1:
                → MUST choose: ask_question
                
            ELSE IF conversation_turns > 1 AND conversation_turns <= 4:
                IF record NOT retrieved yet:
                    → MUST choose: retrieve_interview_record
                ELSE IF record retrieved:
                    → MUST choose: ask_question
                    
            ELSE IF conversation_turns >= 5:
                IF record NOT retrieved yet:
                    → MUST choose: retrieve_interview_record
                ELSE IF saturation NOT evaluated yet:
                    → MUST choose: evaluate_saturation
                ELSE IF saturation_score > 0.8:
                    → MUST choose: generate_user_requirements
                ELSE:
                    → MUST choose: ask_question
            
            CRITICAL RULES:
            1. You MUST follow the IF-ELSE logic above
            2. Check conversation_turns value: {conversation_turns}
            3. From turn 2+: ALWAYS retrieve record before asking
            4. From turn 5+: ALWAYS evaluate saturation after retrieving
            5. If saturation > 0.8: MUST generate requirements

            YOUR DECISION (JSON only):
            {{
                "rationale": "Based on turn {conversation_turns} and current state, I must...",
                "action": "exactly_one_action_from_above"
            }}"""
        return prompt


