import json
from services.kafka_service import KafkaService
from services.minio_service import MinioService
from utils.common import now_iso, make_id
from openai import OpenAI
from typing import Dict, Any
import uuid
from agents.base_agent.action import ActionModule
from agents.archivist_agent.memory import ArchivistMemory
from agents.archivist_agent.profile import ArchivistProfile

class ArchivistAction(ActionModule):

    def __init__(self, publisher: KafkaService, storage_client: MinioService, profile: ArchivistProfile, memory: ArchivistMemory, llm: OpenAI):
        self.publisher = publisher
        self.storage = storage_client
        self.memory = memory
        self.profile = profile
        self.llm = llm
        
    def execute(self, decision: Dict[str, Any], message: dict) -> Dict[str, Any]:
        """Execute the action from thinking module decision."""
        
        action_type = decision.get("action")
        rationale = decision.get("rationale", "")
        
        print(f"[Action] Executing '{action_type}' - Rationale: {rationale}")
        
        # Route to appropriate action handler
        if action_type == "generate_software_requirements_specification":
            return self.generate_software_requirements_specification_action(message, decision)
        else:
            print(f"[Action] Unknown action type: {action_type}")
            return {
                "status": "error",
                "reason": f"unknown_action_{action_type}"
            }
    
    def generate_software_requirements_specification_action(self, message: dict, decision: dict) -> Dict[str, Any]:
        """
        Generate software requirements specification from system requirements and requirement model.
        Store in memory.
        """

        # Get system requirements and requirements model from message
        data = self.retrieve_system_requirements_list_and_requirements_model(message)
        system_requirements_content = data.get("data", {}).get("system_requirements", "")
        requirements_model_content = data.get("data", {}).get("requirements_model", "")

        # Extract rationale for generation
        rationale = decision.get("rationale", "")
        
        prompt = f"""
        INSTRUCTION:
        {rationale}

        Use the following information to inform your generation:

        SYSTEM REQUIREMENTS:
        {system_requirements_content}

        REQUIREMENTS MODEL:
        {requirements_model_content}

        IMPORTANT:
        No asking questions back. Just generate the document as instructed.
        """

        try:
            response = self.llm.chat.completions.create(
                model="gpt-5-nano",
                messages=[{"role": "system", "content": self.profile.system_prompt()},
                          {"role": "user", "content": prompt}]
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Action] Error generating response: {e}")
            return {
                "status": "error",
                "reason": "llm_failure"
            }

        # Store software requirements specification in artifact pool (MinIO) and memory
        artifact_id = f"artifacts/software-requirements-specification/software_requirements_specification_{make_id()}.txt"
        try:
            self.storage.put_object(
                "iredev-application",
                artifact_id,
                answer.encode('utf-8')
            )
            print(f"[Action] Software requirements specification stored in MinIO with artifact ID: {artifact_id}")

        except Exception as e:
            print(f"[Action] Error storing software requirements specification: {e}")
            return {
                "status": "error",
                "reason": "storage_failure"
            }

        # Publish event to Kafka
        self.publisher.publish(topic="artifact_events", message={
            "message_id": str(uuid.uuid4()),
            "artifact_type": "software_requirements_specification",
            "artifact_key": artifact_id
        })

        return {
            "status": "complete"
        }

    def retrieve_system_requirements_list_and_requirements_model(self, message: dict) -> Dict[str, Any]:
        """
        Retrieve system requirements list and requirements model from MinIO.
        """
        # Bucket and object keys
        bucket = "iredev-application"
        system_requirements_key = message.get("system_requirements_list_file_name", "artifacts/system-requirements-list/System Requirements List.txt")
        requirements_model_key = message.get("requirements_model_file_name", "artifacts/requirements-model/Requirements Model.txt")

        try:
            system_requirements_data = self.storage.get_object(bucket, system_requirements_key)
            requirements_model_data = self.storage.get_object(bucket, requirements_model_key)
            system_requirements = system_requirements_data.decode('utf-8')
            requirements_model = requirements_model_data.decode('utf-8')

            print(f"[Action] Data retrieved from MinIO: \nSystem Requirements: {system_requirements[:100]} \nRequirements Model: {requirements_model[:100]}")
            
            return {
                "data": {
                    "system_requirements": system_requirements,
                    "requirements_model": requirements_model
                }
            }

        except Exception as e:
            print(f"[Action] Error retrieving data: {e}")
            return {
                "data": {
                    "system_requirements": "",
                    "requirements_model": ""
                }
            }