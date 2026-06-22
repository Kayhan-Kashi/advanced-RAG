from .base_event import BaseEvent


class PromptAnswerCompletedEvent(BaseEvent):
    conversation_id: str
    dialogue_id: str
    prompt: str
    full_answer: str
    
    @classmethod
    def event_type(cls) -> str:
        return "prompt_answer_completed"
    
    @classmethod
    def topic(cls) -> str:
        return "prompt-answer-completed"