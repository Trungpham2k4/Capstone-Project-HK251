

# services/kafka_service.py
from kafka import KafkaProducer, KafkaConsumer
import json
import threading
from utils.common import now_iso, make_id


class KafkaService:
    def __init__(self, brokers: list[str]):
        self.brokers = brokers
        self.producer = KafkaProducer(
            bootstrap_servers=brokers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8")
        )

    def publish(self, topic: str, message: dict):
        self.producer.send(topic, message)
        self.producer.flush()
        print(f"[KafkaService] Published to {topic}")

    def listen(self, topics: list[str], on_message, group_id: str):
        """Listen on topics in a separate thread and call on_message for each."""
        def loop():
            consumer = KafkaConsumer(
                *topics,
                bootstrap_servers=self.brokers,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset="latest",
                enable_auto_commit=True,
                group_id=group_id,  # bỏ _consumer nếu không cần phân biệt
            )
            for msg in consumer:
                # print(f"[KafkaService] Received on {msg.topic}: {msg.value}")
                on_message(msg.value)

        t = threading.Thread(target=loop, daemon=True)
        t.start()
        print(f"[KafkaService] Listening on {topics}")