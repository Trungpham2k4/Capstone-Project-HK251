import os
from agents.interviewer_agent.interviewer_agent import InterviewerAgent
from agents.enduser_agent.enduser_agent import EndUserAgent
from agents.analyst_agent.analyst_agent import AnalystAgent
from agents.archivist_agent.archivist_agent import ArchivistAgent
import time

from services.kafka_service import KafkaService
from services.minio_service import MinioService
from openai import OpenAI
from dotenv import load_dotenv
from utils.common import now_iso

def build_flow():
    """Simple test flow with real user input via Kafka."""

    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")

    kafka_service = KafkaService(brokers=["localhost:9092"])
    minio_service = MinioService(endpoint="localhost:9000")
    llm_client = OpenAI(api_key=key)

    print("\n" + "="*70)
    print("  REQUIREMENTS ELICITATION INTERVIEW SYSTEM")
    print("="*70 + "\n")

    # Create agents
    print("[Flow] Creating agents...")
    interviewer = InterviewerAgent(
        kafka_service=kafka_service, 
        minio_service=minio_service, 
        llm=llm_client
    )
    enduser = EndUserAgent(
        kafka_service=kafka_service, 
        minio_service=minio_service, 
        llm=llm_client
    )
    analyst = AnalystAgent(
        kafka_service=kafka_service,
        minio_service=minio_service,
        llm=llm_client
    )
    archivist = ArchivistAgent(
        kafka_service=kafka_service,
        minio_service=minio_service,
        llm=llm_client
    )

    # Start agents
    print("[Flow] Starting agents...")
    interviewer.start()
    enduser.start()
    analyst.start()
    archivist.start()

    # Wait for agents to connect to Kafka
    print("[Flow] Waiting for Kafka connections...")
    time.sleep(10)
    
    print("\n" + "-"*70)
    print("  STARTING INTERVIEW")
    print("-"*70 + "\n")

    # Simulate real user input - simple project requirement
    user_input = "I need a currency converter webpage"
    conversation_id = "interview_session_004"
    
    print(f"[User Input] {user_input}")
    print(f"[Conversation ID] {conversation_id}\n")
    
    # Create initial message from user input
    initial_message = {
        "type": "UserInput",
        "sent_from": "User",
        "sent_to": "Interviewer",
        "content": user_input,
        "role": "User",
        "state": "created",
        "conversation_id": conversation_id,
        "timestamp": now_iso()
    }
    
    # Publish to interviewer topic
    # The interviewer will process this and start asking questions
    print("[Flow] Publishing user input to interviewer...")
    kafka_service.publish("user_interviewer", initial_message)


    print("="*70)
    print("  INTERVIEW IN PROGRESS")
    print("="*70)
    print("\nMonitoring conversation... (Press Ctrl+C to stop)\n")
    
    try:
        # Keep running to observe the conversation
        while True:
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("  INTERVIEW STOPPED")
        print("="*70)
        print("\n📁 Check MinIO buckets for generated artifacts")
        print("\n")

if __name__ == "__main__":
    build_flow()