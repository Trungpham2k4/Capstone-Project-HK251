from services.kafka_service import KafkaService
from services.minio_service import MinioService
from utils.common import now_iso, make_id
from openai import OpenAI
from typing import Dict, Any, Optional
import json

from agents.base_agent.action import ActionModule

class InterviewerAction(ActionModule):

    def __init__(self, publisher: KafkaService, 
                 storage_client: MinioService, llm: OpenAI):
        self.publisher = publisher
        self.storage = storage_client
        self.llm = llm
        self.max_iterations = 100
        self.current_iteration = 0
        
    def execute(self, decision: Dict[str, Any], message: dict) -> Dict[str, Any]:
        """Execute the action from thinking module decision."""
        self.current_iteration += 1
        
        # Check max iterations
        if self.current_iteration >= self.max_iterations:
            return {
                "status": "complete",
                "reason": "max_iterations_reached",
                "message": "Process terminated: maximum iterations reached"
            }
        
        action_type = decision.get("action")
        rationale = decision.get("rationale", "")
        
        print(f"[Action] Executing '{action_type}' - Rationale: {rationale}")
        
        # Route to appropriate action handler
        if action_type == "ask_question":
            return self.ask_question_action(message, decision)
        elif action_type == "generate_user_requirements":
            self.reset_iteration_counter()
            return self.generate_requirements_action(message, decision)
        elif action_type == "evaluate_saturation":
            return self.evaluate_saturation_action(message, decision)
        elif action_type == "retrieve_interview_record":
            return self.retrieve_interview_record_action(message, decision)
        else:
            self.reset_iteration_counter()
            print(f"[Action] Unknown action type: {action_type}")
            return {
                "status": "error",
                "reason": f"unknown_action_{action_type}"
            }
    
    def _append_to_interview_record(self, message: dict, content: str, role: str):
        """
        Internal method to append conversation turn to interview record.
        Called by ask_question and respond actions.
        """
        bucket = "interview-records"
        conv_key = message.get("conversation_id", "default_conversation")
        record_key = f"{conv_key}_record.txt"
        
        # Read existing record
        try:
            existing_data = self.storage.get_object(bucket, record_key)
            existing_text = existing_data.decode('utf-8')
        except Exception:
            existing_text = ""
        
        # Format new turn in plain text
        timestamp = now_iso()
        new_turn = f"[{timestamp}] {role}: {content}\n"
        
        # Append to existing record
        updated_text = existing_text + new_turn
        
        # Write back to MinIO
        self.storage.put_object(bucket, record_key, updated_text.encode('utf-8'))
        
        print(f"[Action] Appended to record: {record_key}")
        
        return record_key
    
    def ask_question_action(self, message: dict, decision: dict) -> Dict[str, Any]:
        """
        Interviewer asks a question to EndUser.
        Automatically appends to interview record.
        """
        if message.get("sent_from") != "Enduser" or message.get("sent_to") != "Interviewer":
            # Initial question case
            context = message.get("content", "")
        else:
            # Follow-up question based on enduser response
            context = f"Previous answer from enduser: {message.get('content', '')}"
        
        # Build prompt for question generation
        prompt = f"""You are an experienced requirements interviewer.

Context: {context}

Based on the context and the rationale: "{decision.get('rationale', '')}"

Clarify the user's needs if there is a question from the end user (1-2 sentences).

Generate a single, clear, open-ended question to ask the end user to elicit more requirements.
The question should be conversational and encouraging.

Return ONLY the question text, nothing else."""

        try:
            response = self.llm.chat.completions.create(
                model="gpt-5-nano",
                messages=[{"role": "user", "content": prompt}]
            )
            question = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Action] Error generating question: {e}")
            question = "Could you tell me more about your requirements?"
        
        # Append to interview record
        self._append_to_interview_record(message, question, "Interviewer")
        
        # Create message for the question
        message = self._make_message(
            role="Interviewer",
            message_type="Question",
            content=question,
            sent_from="Interviewer",
            sent_to="Enduser",
            conversation_id=message.get("conversation_id", "default_conversation")
        )
        
        print(f"[Action] Asked question: {question}")
        # Publish to Kafka
        self.publisher.publish("interviewer_enduser", message)
        
        
        
        return {
            "status": "complete",
            "action": "ask_question",
            "message_id": message["message_id"],
            "message": "Question sent, waiting for response"
        }
    
    def retrieve_interview_record_action(self, message: dict, decision: dict) -> Dict[str, Any]:
        """
        Retrieve full conversation record from MinIO.
        Returns data structure compatible with ThinkingModule expectations.
        """
        bucket = "interview-records"
        conv_key = message.get("conversation_id", "default_conversation")
        record_key = f"{conv_key}_record.txt"
        
        try:
            data = self.storage.get_object(bucket, record_key)
            record_text = data.decode('utf-8')

            print("[Action] Data retrieved from MinIO: ", record_text)
            
            # Count turns (each turn has Interviewer and Enduser lines)
            interviewer_count = record_text.count("Interviewer:")
            enduser_count = record_text.count("Enduser:")
            turns = min(interviewer_count, enduser_count)
            
            print(f"[Action] Retrieved record: {turns} turns")
            
            return {
                "status": "continue",
                "action": "retrieve_interview_record",
                "data": {
                    "record_text": record_text,
                    "total_turns": turns,
                    "conversation_id": conv_key
                }
            }
        except Exception as e:
            print(f"[Action] Error retrieving record: {e}")
            return {
                "status": "continue",
                "action": "retrieve_interview_record",
                "data": {
                    "record_text": "",
                    "total_turns": 0,
                    "conversation_id": conv_key
                }
            }
    
    def generate_requirements_action(self, message: dict, decision: dict) -> Dict[str, Any]:
        """
        Generate User Requirements List from conversation.
        Output in plain text format.
        """
        # First retrieve the conversation
        record_result = self.retrieve_interview_record_action(message, decision)
        record_text = record_result.get("data", {}).get("record_text", "")
        
        if not record_text:
            return {
                "status": "error",
                "reason": "no_conversation_found"
            }
        
        prompt = f"""You are a requirements analyst. Analyze this interview conversation and extract user requirements.

Interview Record:
{record_text}

Generate a User Requirements List in plain text format following this structure:

USER REQUIREMENTS LIST
Generated: {now_iso()}
Conversation ID: {message.get("conversation_id", "unknown")}

REQUIREMENTS:

REQ-001: [Brief title]
Description: [Detailed description of the requirement]
Priority: [High/Medium/Low]
Source: [Who mentioned it]
Category: [Functional/Non-functional]

REQ-002: [Brief title]
Description: [Detailed description]
Priority: [High/Medium/Low]
Source: [Who mentioned it]
Category: [Functional/Non-functional]

[Continue for all requirements...]

SUMMARY:
- Total Requirements: [number]
- High Priority: [number]
- Medium Priority: [number]
- Low Priority: [number]

Extract all distinct requirements mentioned. Return ONLY the plain text document."""

        try:
            response = self.llm.chat.completions.create(
                model="gpt-5-nano",
                messages=[{"role": "user", "content": prompt}]
            )
            requirements_text = response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"[Action] Error generating requirements: {e}")
            requirements_text = f"ERROR: Failed to generate requirements\n{str(e)}"
        
        # Store in MinIO as plain text
        bucket = "requirements-artifacts"
        key = f"user_requirements_{make_id()}.txt"
        self.storage.put_object(bucket, key, requirements_text.encode('utf-8'))
        
        # Count requirements (simple heuristic)
        req_count = requirements_text.count("REQ-")
        
        print(f"[Action] Generated requirements: {req_count} items")
        print(f"[Action] Stored at: {bucket}/{key}")
        
        return {
            "status": "complete",
            "action": "generate_user_requirements",
            "artifact_key": key,
            "requirements_count": req_count,
            "bucket": bucket
        }
    
    def evaluate_saturation_action(self, message: dict, decision: dict) -> Dict[str, Any]:
        """
        Evaluate conversation saturation to decide if more questions needed.
        Returns data structure compatible with ThinkingModule expectations.
        """
        # Retrieve conversation
        record_result = self.retrieve_interview_record_action(message, decision)
        record_text = record_result.get("data", {}).get("record_text", "")
        total_turns = record_result.get("data", {}).get("total_turns", 0)
        
        if total_turns < 3:
            return {
                "status": "continue",
                "action": "evaluate_saturation",
                "data": {
                    "saturation_score": 0.2,
                    "recommendation": "continue_interview",
                    "reasoning": "Too few turns for saturation assessment"
                }
            }
        
        # Get last exchanges for analysis
        lines = record_text.strip().split('\n')
        recent_lines = lines[-20:] if len(lines) > 20 else lines
        recent_conversation = '\n'.join(recent_lines)
        
        prompt = f"""Analyze this interview conversation for saturation (repetitive information).

Recent conversation:
{recent_conversation}

Rate saturation from 0.0 (no repetition, rich info) to 1.0 (highly repetitive).
Consider:
- Are new details still emerging?
- Are answers becoming repetitive?
- Is the conversation still productive?

Return ONLY a JSON object:
{{
  "saturation_score": 0.X,
  "recommendation": "continue_interview" or "conclude_interview",
  "reasoning": "brief explanation"
}}"""

        try:
            response = self.llm.chat.completions.create(
                model="gpt-5-nano",
                messages=[{"role": "user", "content": prompt}]
            )
            result_text = response.choices[0].message.content.strip()
            
            # Clean and parse JSON
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(result_text)
            saturation_score = result.get("saturation_score", 0.5)
            reasoning = result.get("reasoning", "No reasoning provided")
            
        except Exception as e:
            print(f"[Action] Error evaluating saturation: {e}")
            saturation_score = 0.5
            reasoning = "Error in evaluation, defaulting to continue"
        
        print(f"[Action] Saturation score: {saturation_score}")
        
        # Return data structure that ThinkingModule expects
        return {
            "status": "continue",
            "action": "evaluate_saturation",
            "data": {
                "saturation_score": saturation_score,
                "reasoning": reasoning
            }
        }