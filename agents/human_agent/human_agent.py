import json
from typing import Dict
from agents.human_agent.action import HumanInterventionManager
import threading


class HumanSupervisorCLI:
    def __init__(self, approval_required: bool = True):
        self.human_manager = HumanInterventionManager(
            approval_required=approval_required
        )
        self.supervisor_thread = None
        self.is_running = False
        self.responses = {}
        self.done_event = threading.Event()

    def start(self):
        self.is_running = True
        self.supervisor_thread = threading.Thread(
            target=self.start_supervisor_interface
        )
        self.supervisor_thread.daemon = True
        self.supervisor_thread.start()
        print("Starting supervisor interface in separate thread")
        self.done_event.wait()

    def stop(self):
        """Stop the intervention listener"""
        self.is_running = False
        if self.supervisor_thread:
            self.supervisor_thread.join(timeout=5)

    def start_supervisor_interface(self):
        """Start the human supervisor command line interface"""
        print("👤 HUMAN SUPERVISOR INTERFACE")
        print("Commands: list, approve <id>, deny <id>, exit")
        print("-" * 50)

        while self.is_running:
            try:
                command = input("\nSupervisor> ").strip()

                if command.lower() in ["exit", "quit"]:
                    break
                elif command.lower() == "list":
                    self._list_pending_requests()
                elif command.startswith("approve"):
                    self._process_approval_command(command)
                    break
                elif command.startswith("deny"):
                    self._process_denial_command(command)
                    break
                # elif command.startswith("modify"):
                #     self._process_modify_command(command)
                else:
                    print("Unknown command. Use: list, approve <id>, deny <id>")

            except KeyboardInterrupt:
                break
            except Exception as e:
                raise
                print(f"Error: {e}")
        self.done_event.set()

    def _list_pending_requests(self):
        """List all pending approval requests"""
        print("\n📋 PENDING APPROVAL REQUESTS:")
        print("-" * 40)

        for i, request in enumerate(self.human_manager.pending_approvals):
            print(f"{i}. [{request['type'].value}] ID: {request['id']}")
            print(
                f"   Data: {request['data'].get('content', request['data'].get('response', 'N/A'))[:100]}..."
            )
            print(f"   Time: {request['timestamp'].strftime('%H:%M:%S')}")

    def _process_approval_command(self, command: str):
        """Process approval command"""
        parts = command.split()
        if len(parts) < 2:
            print("Usage: approve <request_id>")
            return

        request_id = parts[1]
        custom_data = None

        # Check for modification data
        # if len(parts) > 2 and parts[2] == "modify":
        #     custom_data = self._get_modification_data()

        success = self.human_manager.process_human_decision(
            request_id, "approve", custom_data
        )
        if success:
            print(f"✅ Request {request_id} approved")
        else:
            print(f"❌ Failed to approve {request_id}")

    def _process_denial_command(self, command: str):
        """Process denial command"""
        parts = command.split()
        if len(parts) < 2:
            print("Usage: deny <request_id>")
            return

        request_id = parts[1]
        custom_data = self._get_modification_data()
        success = self.human_manager.process_human_decision(
            request_id, "deny", custom_data
        )

        if success:
            print(f"❌ Request {request_id} denied")
        else:
            print(f"⚠️ Failed to process denial for {request_id}")

    def _process_modify_command(self, command: str):
        """Process modify command"""
        parts = command.split()
        if len(parts) < 2:
            print("Usage: modify <request_id>")
            return

        request_id = parts[1]
        custom_data = self._get_modification_data()

        success = self.human_manager.process_human_decision(
            request_id, "modify", custom_data
        )

        if success:
            print(f"✏️ Request {request_id} modified and approved")
        else:
            print(f"⚠️ Failed to modify {request_id}")

    def _get_modification_data(self) -> Dict:
        """Get feedback data from human supervisor"""
        print("Enter feedback data (or 'skip' for no feedback):")
        # try:
        data_input = input("Feedback> ").strip()
        if data_input.lower() == "skip":
            return ""
        return data_input
        # except json.JSONDecodeError:
        #     print("Invalid JSON format")
        #     return {}
