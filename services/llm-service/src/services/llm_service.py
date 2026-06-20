# llm-service/src/services/llm_service.py
import os
import logging
from injector import inject
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from typing import List, Dict, Any, Optional
from src.services.rag_service import RagService

logger = logging.getLogger(__name__)


class LLMService:
    """Service for RAG-enhanced LLM operations with HyDE support"""
    
    @inject
    def __init__(self, rag_service: RagService):
        self.rag_service = rag_service
        
        # Initialize LLM
        self.llm = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "gemma3:12b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.3")),
        )
        
        # ============ HYDE PROMPT (LLM Call 1) ============
        self.hyde_prompt = ChatPromptTemplate.from_template("""
You are an AI assistant. Given a user question, write a hypothetical document that would contain the answer.
This hypothetical document will be used for semantic search to find similar real documents.

Write a clear, detailed, and informative passage that answers the question. The passage should be 3-5 sentences long.

User Question: {question}

Hypothetical Document:
""")
        
        # RAG Prompt with history support and source citation
        self.prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the user's question based ONLY on the provided context and previous conversation history.
If the answer is not in the context or in the conversation history, say that you don't know.

IMPORTANT: When you use information from the context, cite the source using [Filename, Page X] format at the end of each sentence or paragraph that uses that source.

Example: "The company reported revenue of $10M in Q4 2024 [Annual_Report.pdf, Page 5]."

<conversation_history>
{history}
</conversation_history>

<context>
{context}
</context>

Question: {input}

Answer: 
""")
        
        # Environment variable to control HyDE (default: True)
        self.use_hyde_default = os.getenv("USE_HYDE", "False").lower() == "true"
        
        logger.info("✅ LLMService initialized with HyDE support")
        logger.info(f"   HyDE default: {self.use_hyde_default}")

    def generate_hypothetical_document(self, query: str) -> str:
        """
        LLM Call 1: Generate a hypothetical document from the query (HyDE).
        This is called BEFORE retrieval to transform the query.
        
        Args:
            query: User question
        
        Returns:
            Hypothetical document string (used as the search query)
        """
        try:
            logger.info("=" * 60)
            logger.info("📄 [LLM Call 1 - HyDE] Generating hypothetical document...")
            logger.info(f"   Original query: {query[:100]}...")
            
            messages = self.hyde_prompt.format_messages(question=query)
            response = self.llm.invoke(messages)
            hyde_doc = response.content.strip()
            
            logger.info(f"   ✅ HyDE generated: {hyde_doc[:150]}...")
            logger.info("=" * 60)
            return hyde_doc
            
        except Exception as e:
            logger.error(f"HyDE generation failed: {e}")
            return query  # Fallback to original query
    
    async def generate_hypothetical_document_async(self, query: str) -> str:
        """
        LLM Call 1: Async version for generating hypothetical document.
        Called BEFORE retrieval.
        
        Args:
            query: User question
        
        Returns:
            Hypothetical document string (used as the search query)
        """
        try:
            logger.info("=" * 60)
            logger.info("📄 [LLM Call 1 - HyDE Async] Generating hypothetical document...")
            logger.info(f"   Original query: {query[:100]}...")
            
            messages = self.hyde_prompt.format_messages(question=query)
            response = await self.llm.ainvoke(messages)
            hyde_doc = response.content.strip()
            
            logger.info(f"   ✅ HyDE generated: {hyde_doc[:150]}...")
            logger.info("=" * 60)
            return hyde_doc
            
        except Exception as e:
            logger.error(f"HyDE generation failed: {e}")
            return query  # Fallback to original query

    async def generate(
        self, 
        prompt: str, 
        file_ids: Optional[List[str]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        use_hyde: Optional[bool] = None
    ) -> str:
        """
        Generate answer using RAG with optional HyDE.
        
        Flow:
        1. If use_hyde=True: LLM Call 1 - Generate HyDE query
        2. Retrieve using HyDE query (or original query if no HyDE)
        3. LLM Call 2 - Generate final answer with context
        
        Args:
            prompt: User question
            file_ids: Optional list of file IDs to filter by
            history: Optional conversation history as list of {role, content} dicts
            use_hyde: Whether to use HyDE (defaults to env setting)
        
        Returns:
            Generated answer with source citations (PDF filename and page number)
        """
        # Use default from env if not specified
        if use_hyde is None:
            use_hyde = self.use_hyde_default
        
        try:
            logger.info("=" * 80)
            logger.info(f"📝 User Question: {prompt}")
            logger.info(f"   HyDE: {use_hyde}")
            if file_ids:
                logger.info(f"   📁 Filtering to file_ids: {file_ids}")
            if history:
                logger.info(f"   📜 History: {len(history)} messages")
            logger.info("=" * 80)
            
            # ============ STEP 1: Generate HyDE query (LLM Call 1) ============
            search_query = prompt
            
            if use_hyde:
                logger.info("\n🔄 [STEP 1] Generating HyDE query (LLM Call 1)...")
                search_query = self.generate_hypothetical_document(prompt)
                logger.info(f"   ✅ Using HyDE query for retrieval: {search_query[:150]}...")
            else:
                logger.info("\n📝 [STEP 1] Using original query (no HyDE)")
            
            # ============ STEP 2: Retrieve with the query ============
            logger.info("\n🔍 [STEP 2] Retrieving chunks...")
            retrieved_chunks = self.rag_service.retrieve(
                query=search_query,  # Use HyDE query if available
                k=10,
                file_ids=file_ids
            )
            
            if not retrieved_chunks:
                logger.warning("❌ No chunks retrieved")
                if file_ids:
                    return f"I don't have any relevant information in the selected documents to answer: '{prompt}'."
                return "I don't have any relevant information to answer this question."
            
            # Store HyDE metadata on chunks
            if use_hyde:
                for chunk in retrieved_chunks:
                    chunk.metadata['hyde_used'] = True
                    chunk.metadata['hyde_query'] = search_query[:500]
                    chunk.metadata['original_query'] = prompt[:200]
            
            logger.info(f"   ✅ Retrieved {len(retrieved_chunks)} chunks")
            
            # ============ STEP 3: Build context ============
            logger.info("\n📚 [STEP 3] Building context...")
            context_parts = []
            for chunk in retrieved_chunks:
                filename = chunk.metadata.get('filename', 'Unknown.pdf')
                page_num = chunk.metadata.get('page_number', 'Unknown')
                content = chunk.page_content
                
                context_parts.append(
                    f"[Source: {filename}, Page: {page_num}]\n{content}"
                )
            
            context_text = "\n\n---\n\n".join(context_parts)
            
            # Log retrieved chunks with page info
            logger.info(f"📚 Retrieved {len(retrieved_chunks)} chunks:")
            for i, chunk in enumerate(retrieved_chunks[:5], 1):
                chunk_preview = chunk.page_content[:150].replace('\n', ' ')
                filename = chunk.metadata.get('filename', 'unknown')
                page_num = chunk.metadata.get('page_number', 'N/A')
                score = chunk.metadata.get('reranker_score', 'N/A')
                hyde_used = chunk.metadata.get('hyde_used', False)
                logger.info(f"   {i}. [Page {page_num}] Score={score} - {chunk_preview}... (from: {filename}) {'[HyDE]' if hyde_used else ''}")
            
            # ============ STEP 4: Build history ============
            history_text = ""
            if history:
                history_lines = []
                for msg in history:
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    if role == 'user':
                        history_lines.append(f"User: {content}")
                    else:
                        history_lines.append(f"Assistant: {content}")
                history_text = "Previous conversation:\n" + "\n".join(history_lines) + "\n"
                logger.info(f"📜 Added {len(history)} history messages to prompt")
            
            # ============ STEP 5: LLM Call 2 - Generate final answer ============
            logger.info("\n🤖 [STEP 4 - LLM Call 2] Generating final answer...")
            formatted_messages = self.prompt.format_messages(
                context=context_text,
                input=prompt,  # Use original prompt for the answer
                history=history_text
            )
            
            # Log prompt summary
            logger.info("=" * 80)
            logger.info(f"📊 PROMPT SUMMARY {'(HyDE)' if use_hyde else '(Standard)'}")
            logger.info("=" * 80)
            logger.info(f"   - Chunks in context: {len(retrieved_chunks)}")
            logger.info(f"   - Context length: {len(context_text)} characters")
            logger.info(f"   - History messages: {len(history) if history else 0}")
            logger.info(f"   - Query length: {len(prompt)} characters")
            if use_hyde:
                logger.info(f"   - HyDE query length: {len(search_query)} characters")
            if file_ids:
                logger.info(f"   - Filtering to: {file_ids}")
            
            # Log page distribution
            pages_used = {}
            for chunk in retrieved_chunks:
                page = chunk.metadata.get('page_number', 'unknown')
                filename = chunk.metadata.get('filename', 'unknown')
                key = f"{filename}:{page}"
                pages_used[key] = pages_used.get(key, 0) + 1
            logger.info(f"   - Pages used: {pages_used}")
            
            # Log full prompt (truncated for readability)
            logger.info("-" * 80)
            logger.info("📝 PROMPT CONTENT:")
            logger.info("-" * 80)
            prompt_content = formatted_messages[0].content
            if len(prompt_content) > 1500:
                logger.info(f"{prompt_content[:1500]}...\n[TRUNCATED - total {len(prompt_content)} chars]")
            else:
                logger.info(prompt_content)
            logger.info("=" * 80)
            
            # ============ STEP 6: Generate response ============
            response = await self.llm.ainvoke(formatted_messages)
            answer = response.content.strip()
            
            # ============ STEP 7: Add source summary if not already cited ============
            if "[Source:" not in answer and "[" not in answer:
                sources = []
                seen = set()
                for chunk in retrieved_chunks[:5]:
                    filename = chunk.metadata.get('filename', 'Unknown.pdf')
                    page_num = chunk.metadata.get('page_number', 'N/A')
                    key = f"{filename}:{page_num}"
                    if key not in seen:
                        seen.add(key)
                        sources.append(f"  • {filename} (Page {page_num})")
                
                if sources:
                    answer += f"\n\n**Sources:**\n" + "\n".join(sources)
            
            logger.info(f"\n💬 Response: {answer[:300]}...")
            return answer
            
        except Exception as e:
            logger.error(f"❌ Generation error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"Error generating response: {str(e)}"
    
    async def generate_with_sources(
        self, 
        prompt: str, 
        file_ids: Optional[List[str]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        use_hyde: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Generate answer and return with source documents including PDF filename and page numbers.
        
        Args:
            prompt: User question
            file_ids: Optional list of file IDs to filter by
            history: Optional conversation history
            use_hyde: Whether to use HyDE (defaults to env setting)
        
        Returns:
            Dictionary with answer, sources (with filename and page numbers)
        """
        if use_hyde is None:
            use_hyde = self.use_hyde_default
        
        try:
            logger.info(f"📝 User Question: {prompt}")
            logger.info(f"   HyDE: {use_hyde}")
            if file_ids:
                logger.info(f"   📁 Filtering to file_ids: {file_ids}")
            if history:
                logger.info(f"   📜 History: {len(history)} messages")
            
            # Generate answer
            answer = await self.generate(
                prompt=prompt,
                file_ids=file_ids,
                history=history,
                use_hyde=use_hyde
            )
            
            # Get chunks for sources
            search_query = prompt
            if use_hyde:
                search_query = self.generate_hypothetical_document(prompt)
            
            retrieved_chunks = self.rag_service.retrieve(
                query=search_query,
                k=10,
                file_ids=file_ids
            )
            
            # Build sources
            sources = []
            for i, chunk in enumerate(retrieved_chunks[:10], 1):
                source = {
                    "rank": i,
                    "content_preview": chunk.page_content[:200],
                    "full_content": chunk.page_content,
                    "document_id": chunk.metadata.get('document_id', ''),
                    "filename": chunk.metadata.get('filename', 'Unknown.pdf'),
                    "page_number": chunk.metadata.get('page_number', 'Unknown'),
                    "reranker_score": chunk.metadata.get('reranker_score'),
                    "chunk_index": chunk.metadata.get('chunk_index'),
                    "hyde_used": chunk.metadata.get('hyde_used', False)
                }
                sources.append(source)
            
            return {
                "answer": answer,
                "sources": sources,
                "total_chunks_retrieved": len(retrieved_chunks),
                "used_hyde": use_hyde,
                "hyde_document": search_query if use_hyde else None
            }
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "sources": [],
                "error": str(e)
            }
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics"""
        return self.rag_service.get_stats()