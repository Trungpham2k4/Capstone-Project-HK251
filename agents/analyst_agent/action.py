import json
from services.kafka_service import KafkaService
from services.minio_service import MinioService
from utils.common import now_iso, make_id
from openai import OpenAI
from typing import Dict, Any

from agents.base_agent.action import ActionModule
from agents.analyst_agent.memory import AnalystMemory
from agents.analyst_agent.profile import AnalystProfile

class AnalystAction(ActionModule):

    def __init__(self, publisher: KafkaService, storage_client: MinioService, profile: AnalystProfile, memory: AnalystMemory, llm: OpenAI):
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
        if action_type == "generate_system_requirements":
            return self.generate_system_requirements_action(message, decision)
        elif action_type == "choose_requirement_model":
            return self.choose_requirement_model_action(message, decision)
        elif action_type == "generate_requirement_model":
            return self.generate_requirement_model_action(message, decision)
        else:
            print(f"[Action] Unknown action type: {action_type}")
            return {
                "status": "error",
                "reason": f"unknown_action_{action_type}"
            }
    
    def generate_system_requirements_action(self, message: dict, decision: dict) -> Dict[str, Any]:
        """
        Generate system requirements from user and environment requirements.
        Store in memory.
        """

        # Get user requirements and operating environment list from message
        data = self.retrieve_url_and_oel(message)
        user_requirements_content = data.get("data", {}).get("user_requirements", "")
        operating_environment_content = data.get("data", {}).get("operating_environment", "")

        # Extract rationale for generation
        rationale = decision.get("rationale", "")
        
        prompt = f"""
        Use the following information to inform your generation:

        User Requirements:
        {user_requirements_content}

        Operating Environment:
        {operating_environment_content}

        INSTRUCTION:
        {rationale}

        IMPORTANT:
        No asking questions, only system requirements content

        Generate a system requirements list follow this structure: 
        SYSTEM REQUIREMENTS LIST
        CREATED: {now_iso()}

        <system_requirements_content>
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

        # Store system requirements in artifact pool (MinIO) and memory
        artifact_id = f"analyst-artifacts/system_requirements_{make_id()}.txt"
        try:
            self.storage.put_object(
                "requirements-artifacts",
                artifact_id,
                answer.encode('utf-8')
            )
            print(f"[Action] System requirements stored in MinIO with artifact ID: {artifact_id}")

            # Update memory: content and flag
            self.memory.write("system_requirements", answer)

        except Exception as e:
            print(f"[Action] Error storing system requirements: {e}")
            return {
                "status": "error",
                "reason": "storage_failure"
            }

        # Publish event to Kafka
        self.publisher.publish(topic="artifact_events", message={
            "artifact_type": "system_requirements_list",
            "artifact_key": artifact_id
        })

        return {
            "status": "continue"
        }

    def retrieve_url_and_oel(self, message: dict) -> Dict[str, Any]:
        """
        Retrieve user requirement list and operating environment list from MinIO.
        """
        # Bucket and object keys
        bucket = "requirements-artifacts"
        user_requirements_key = message.get("user_requirements_list_file_name", "User Requirements List.txt")
        operating_environment_key = message.get("operating_environment_list_file_name", "Operating Env List.txt")

        try:
            user_requirements_data = self.storage.get_object(bucket, user_requirements_key)
            operating_environment_data = self.storage.get_object(bucket, operating_environment_key)

            user_requirements = user_requirements_data.decode('utf-8')
            operating_environment = operating_environment_data.decode('utf-8')

            print(f"[Action] Data retrieved from MinIO: \nUser Requirements: {user_requirements[:100]} \nOperating Environment: {operating_environment[:100]}")
            
            return {
                "data": {
                    "user_requirements": user_requirements,
                    "operating_environment": operating_environment
                }
            }

        except Exception as e:
            print(f"[Action] Error retrieving data: {e}")
            return {
                "data": {
                    "user_requirements": "",
                    "operating_environment": ""
                }
            }


    def choose_requirement_model_action(self, message: dict, decision: dict) -> Dict[str, Any]:
        """
        Choose a requirement modeling methodology (e.g., UML, SysML-v2).
        Store choice in memory.
        """
        # Extract rationale for choice
        rationale = decision.get("rationale", "")
        
        prompt = f"""You are to choose an appropriate requirement modeling methodology based on the following instructions:

        {rationale}

        Consider the following system requirements:
        {self.memory.read("system_requirements")[0]}

        OUTPUT FORMAT (strict JSON only):
        {{
            "requirement_model": "<chosen_model>"
        }}

        Where <chosen_model> is one of: "Use case diagram", "SysML-v2 diagram"
        """
        try:
            response = self.llm.chat.completions.create(
                model="gpt-5-nano",
                messages=[{"role": "system", "content": self.profile.system_prompt()},
                          {"role": "user", "content": prompt}]
            )
            print(f"[Action] LLM response for requirement model choice: {response.choices[0].message.content.strip()}")
            answer: dict[str, str] = json.loads(response.choices[0].message.content.strip())
            # Update memory with chosen model
            self.memory.write("requirement_model", answer.get("requirement_model"))

        except Exception as e:
            print(f"[Action] Error generating response: {e}")
            return {
                "status": "error",
                "reason": "llm_failure"
            }

        return {
            "status": "continue"
        }
    
    def generate_requirement_model_action(self, message: dict, decision: dict) -> Dict[str, Any]:
        """
        Generate a requirement model based on system requirements and chosen modeling methodology.
        Store model in MinIO.
        """

        # Extract rationale for generation
        rationale = decision.get("rationale", "")
        
        prompt = f"""You are going to generate a requirement model based on the following the instructions:

        {rationale}

        System Requirements:
        {self.memory.read("system_requirements")[0]}

        Chosen Requirement Model:
        {self.memory.read("requirement_model")[0]}

        MANDATORY DECISION LOGIC - FOLLOW EXACTLY:
        IF Chosen Requirement Model is "Use case diagram":
            → Generate a Use Case Diagram using PlantUML syntax.
        ELSE IF Chosen Requirement Model is "SysML-v2 diagram":
            → Generate a SysML-v2 Diagram using SysML-v2 Pilot syntax.

        IMPORTANT:
        No asking questions, only requirement model content
        Only output the diagram syntax without any extra explanation. YOU MUST FOLLOW THE SYNTAX OF THE SELECTED MODELING LANGUAGE.

        STRUCTURE YOUR OUTPUT AS FOLLOWS:

        REQUIREMENT MODEL
        CREATED: {now_iso()}
        <requirement_model_syntax_diagram> <- syntax diagram here

        """

        try:
            response = self.llm.chat.completions.create(
                model="gpt-5",
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
        
        artifact_id = f"analyst-artifacts/requirement_model_{make_id()}.txt"
        try:
            self.storage.put_object(
                "requirements-artifacts",
                artifact_id,
                answer.encode('utf-8')
            )
            print(f"[Action] Requirement model stored in MinIO with artifact ID: {artifact_id}")

        except Exception as e:
            print(f"[Action] Error storing requirement model: {e}")
            return {
                "status": "error",
                "reason": "storage_failure"
            }
        
        # Publish event to Kafka
        self.publisher.publish("artifact_events", message={
            "artifact_type": "requirements_model",
            "artifact_key": artifact_id
        })
        
        return {
            "status": "complete"
        }