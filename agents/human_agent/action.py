import threading
from datetime import datetime
import uuid
from typing import Dict, Optional

from enum import Enum


class InterventionType(Enum):
    MEMORY_APPROVAL = "memory_approval"
    RESPONSE_APPROVAL = "response_approval"
    MEMORY_CORRECTION = "memory_correction"
    ACTION_APPROVAL = "action_approval"
    URGENT_INTERVENTION = "urgent_intervention"


class HumanInterventionManager:
    def __init__(self, approval_required: bool = True):
        self.approval_required = approval_required
        self.pending_approvals = []
        self.approval_callbacks = {}
        self.intervention_thread = None
        self.is_running = False

    # def start_intervention_listener(self):
    #     """Start listening for human input in a separate thread"""
    #     self.is_running = True
    #     self.intervention_thread = threading.Thread(target=self._process_human_input)
    #     self.intervention_thread.daemon = True
    #     self.intervention_thread.start()
    #     print("Human-in-the-loop listener started...")

    # def _process_human_input(self):
    #     """Process human input in a separate thread"""
    #     while self.is_running:
    #         try:
    #             if not self.pending_approvals.empty():
    #                 # Check for timeouts
    #                 current_time = datetime.now()

    #                 # We'll process input in the main thread for simplicity
    #                 # In production, you'd use proper async I/O
    #                 # time.sleep(1)

    #         except Exception as e:
    #             print(f"Error in intervention listener: {e}")

    def process_human_decision(
        self, approval_id: str, decision: str, custom_data: Optional[str] = None
    ):
        """Process human decision for a pending approval"""
        if approval_id not in self.approval_callbacks:
            print(f"Unknown approval ID: {approval_id}")
            return False

        callback = self.approval_callbacks[approval_id]

        try:
            result = callback(decision, custom_data)
            del self.approval_callbacks[approval_id]
            for approval_elem in self.pending_approvals:
                if approval_elem.id == approval_id:
                    self.pending_approvals.remove(approval_elem)
                    break
            return result
        except Exception as e:
            print(f"Error processing human decision: {e}")
            return False

    def request_approval(
        self,
        item_type: InterventionType,
        item_data: Dict,
        callback: callable,
        timeout: int = 300,
    ):  # 5 minutes default timeout
        """Request human approval for an action/memory"""
        approval_id = str(uuid.uuid4())

        approval_request = {
            "id": approval_id,
            "type": item_type,
            "data": item_data,
            "callback": callback,
            "timestamp": datetime.now(),
            "timeout": timeout,
            "status": "pending",
        }

        self.pending_approvals.append(approval_request)
        self.approval_callbacks[approval_id] = callback

        print(f"\n🔔 HUMAN APPROVAL REQUIRED [{item_type.value}]")
        print(f"ID: {approval_id}")
        # self._display_approval_request(item_type, item_data)

        return approval_id

    def _display_approval_request(self, item_type: InterventionType, item_data: Dict):
        """Display approval request in human-readable format"""
        if item_type == InterventionType.MEMORY_APPROVAL:
            print(f"🧠 MEMORY TO BE STORED:")
            print(f"Content: {item_data['content']}")
            print(f"Type: {item_data['memory_type']}")
            print(f"Importance: {item_data['importance']}")
            if "reason" in item_data:
                print(f"Reason: {item_data['reason']}")

    # def stop_intervention_listener(self):
    #     """Stop the intervention listener"""
    #     self.is_running = False
    #     if self.intervention_thread:
    #         self.intervention_thread.join(timeout=5)
