import asyncio
import json
import traceback
import os
from confluent_kafka import Consumer
from injector import Injector

from src.dependencies import DependencyInjection
from common.message_bus.bus import MessageBus #type: ignore
from common.events import PromptAnswerCompletedEvent #type: ignore
from src.handlers.prompt_answer_completed_handler import PromptAnswerCompletedHandler
from src.database.sqlite_session import get_session


# ============================================
# Kafka settings
# ============================================

KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
CONSUMER_GROUP = os.getenv('KAFKA_CONSUMER_GROUP', 'elearning-worker-group')
INPUT_TOPIC = os.getenv('KAFKA_INPUT_TOPIC', 'prompt-answer-completed')


# ============================================
# Event type registry
# ============================================

EVENT_TYPES = {
    "prompt_answer_completed": PromptAnswerCompletedEvent,
}


# ============================================
# Dependency Injection (services only)
# ============================================

injector = Injector([DependencyInjection()])
bus = MessageBus(injector)

bus.register(PromptAnswerCompletedEvent, PromptAnswerCompletedHandler)


# ============================================
# Core message processor
# ============================================

async def handle_message(message: dict):
    """Process a single Kafka message"""
    try:
        event_type = message.get("event_type")
        
        print(f"[elearning-service] 📩 Received {event_type}", flush=True)
        
        event_class = EVENT_TYPES.get(event_type)
        if not event_class:
            print(f"[elearning-service] ⚠ Unknown event: {event_type}", flush=True)
            return
        
        event = event_class(**message)
        
        # Create DB session per message
        with next(get_session()) as db:
            result = await bus.dispatch(event, db=db)
        
        # If a handler returns a new event, publish it
        if result:
            # E-Learning service doesn't produce events, but if needed
            print(f"[elearning-service] 📤 Would publish: {result}", flush=True)
            
    except Exception as exc:
        print(f"[elearning-service] ❌ Error processing message: {exc}", flush=True)
        traceback.print_exc()


# ============================================
# Kafka consumer setup
# ============================================

async def create_consumer() -> Consumer:
    """Create and configure Kafka consumer"""
    
    conf = {
        'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
        'group.id': CONSUMER_GROUP,
        'auto.offset.reset': 'earliest',
        'enable.auto.commit': True,
        'auto.commit.interval.ms': 5000,
    }
    
    consumer = Consumer(conf)
    consumer.subscribe([INPUT_TOPIC])
    
    print(f"[elearning-service] ✅ Consumer subscribed to: {INPUT_TOPIC}", flush=True)
    print(f"[elearning-service]    Group ID: {CONSUMER_GROUP}", flush=True)
    
    return consumer


# ============================================
# Main worker
# ============================================

async def main():
    print("[elearning-service] 🚀 Starting worker...", flush=True)
    
    # Create consumer
    consumer = await create_consumer()
    
    print(f"[elearning-service] ✅ Worker listening to: {list(EVENT_TYPES.keys())}", flush=True)
    print("[elearning-service] ⏳ Waiting for events...", flush=True)
    
    try:
        while True:
            # Poll for messages (non-blocking)
            msg = consumer.poll(timeout=1.0)
            
            if msg is None:
                await asyncio.sleep(0.1)
                continue
            
            if msg.error():
                print(f"[elearning-service] Consumer error: {msg.error()}", flush=True)
                continue
            
            # Parse and process message
            try:
                message = json.loads(msg.value().decode('utf-8'))
                await handle_message(message)
                
            except json.JSONDecodeError as e:
                print(f"[elearning-service] ❌ Failed to decode message: {e}", flush=True)
                continue
                
    except KeyboardInterrupt:
        print("[elearning-service] ⚠ Worker interrupted", flush=True)
    finally:
        consumer.close()
        print("[elearning-service] ✅ Worker shutdown", flush=True)


if __name__ == "__main__":
    asyncio.run(main())