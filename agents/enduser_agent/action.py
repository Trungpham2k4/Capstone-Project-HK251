from services.kafka_service import KafkaService
from services.minio_service import MinioService
from utils.common import now_iso, make_id
from openai import OpenAI
from typing import Dict, Any

from agents.base_agent.action import ActionModule

class EndUserAction(ActionModule):

    def __init__(self, publisher: KafkaService, 
                 storage_client: MinioService, llm: OpenAI):
        self.publisher = publisher
        self.storage = storage_client
        self.llm = llm
        
    def execute(self, decision: Dict[str, Any], message: dict) -> Dict[str, Any]:
        """Execute the action from thinking module decision."""
        
        action_type = decision.get("action")
        rationale = decision.get("rationale", "")
        
        print(f"[Action] Executing '{action_type}' - Rationale: {rationale}")
        
        # Route to appropriate action handler
        if action_type == "respond" or action_type == "clarify":
            return self.respond_action(message, decision)
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
    
    def respond_action(self, message: dict, decision: dict) -> Dict[str, Any]:
        """
        EndUser responds to interviewer's question.
        Automatically appends to interview record.
        """
        question = message.get("content", "")
        
        prompt = f"""You are an end user being interviewed about software requirements.

The interviewer asked: "{question}"

Based on this rationale: "{decision.get('rationale', '')}"

Provide a realistic, detailed response as an end user would. Include:
- Your needs and pain points
- Specific examples if relevant
- Any constraints or preferences

Keep the response conversational and natural (2-4 sentences).

Return ONLY the response text."""

        try:
            response = self.llm.chat.completions.create(
                model="gpt-5-nano",
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Action] Error generating response: {e}")
            answer = "I need a system that is user-friendly and efficient."
        
        # Append to interview record
        self._append_to_interview_record(message, answer, "Enduser")
        
        # Create message
        message = self._make_message(
            role="Enduser",
            message_type="Response",
            content=answer,
            sent_from="Enduser",
            sent_to="Interviewer",
            conversation_id=message.get("conversation_id", "default_conversation")
        )
        
        print(f"[Action] Responded: {answer}")
        # Publish to Kafka
        self.publisher.publish("enduser_interviewer", message)
        
        return {
            "status": "complete",
            "action": "respond",
            "message_id": message["message_id"],
            "message": "Response sent, waiting for next question"
        }