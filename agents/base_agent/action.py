from typing import Dict, Any

from utils.common import now_iso, make_id

class ActionModule:
    """
    Action module executes actions determined by ThinkingModule.
    Stores data in plain text format in MinIO.
    """
        
    def execute(self, decision: Dict[str, Any], event: dict) -> Dict[str, Any]:
        """Execute the action from thinking module decision."""
        pass

    def _make_message(self, role: str, message_type: str, content: str, 
                      sent_from: str, sent_to: str, conversation_id: str = "default") -> dict:
        """Create a standardized message dictionary."""
        return {
            "message_id": make_id(),
            "type": message_type,
            "role": role,
            "content": content,
            "state": "created",
            "sent_from": sent_from,
            "sent_to": sent_to,
            "conversation_id": conversation_id,
            "timestamp": now_iso()
        }
    
    def reset_iteration_counter(self):
        """Reset iteration counter for new conversation."""
        self.current_iteration = 0