import os

from langchain_ollama import ChatOllama


class LLMService:
    """LLM Service using LangChain with pipe operator (LCEL)"""
    
    def __init__(self):
        self.llm = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "gemma3:12b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.3")),
        )