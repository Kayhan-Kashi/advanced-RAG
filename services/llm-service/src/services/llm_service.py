import os
import logging
from injector import inject
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from typing import List, Dict, Any, Optional
from src.services.rag_service import RagService

logger = logging.getLogger(__name__)

class LLMService:
    """Service for RAG-enhanced LLM operations using Ensemble Retriever only"""
    
    @inject
    def __init__(self, rag_service: RagService):
        self.rag_service = rag_service
        
        # 1. Initialize LLM
        self.llm = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "gemma3:12b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.3")),
        )
        
        # 2. Define RAG Prompt with better instructions
        self.prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant. Answer the user's question based ONLY on the provided context.
        If the answer is not in the context, say that you don't know.
        
        <context>
        {context}
        </context>

        Question: {input}
        
        Answer: 
        """)
        
        # 3. Create document combination chain (kept for potential future use)
        combine_docs_chain = create_stuff_documents_chain(self.llm, self.prompt)
        
        # 4. Create retrieval chain with Ensemble Retriever ONLY
        self.ensemble_retriever = self.rag_service.get_ensemble_retriever(
            faiss_weight=float(os.getenv("FAISS_WEIGHT", "0.7")),
            bm25_weight=float(os.getenv("BM25_WEIGHT", "0.3")),
            faiss_k=int(os.getenv("FAISS_RETRIEVAL_K", "50")),
            bm25_k=int(os.getenv("BM25_RETRIEVAL_K", "50"))
        )
        
        if self.ensemble_retriever is None:
            error_msg = "❌ Ensemble Retriever is not available!"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Store chain for potential fallback, but we'll use direct LLM calls
        self.combine_docs_chain = combine_docs_chain
        self.chain = create_retrieval_chain(self.ensemble_retriever, combine_docs_chain)
        self.retriever_type = "ensemble (FAISS + BM25)"
        
        logger.info("✅ LLM Service initialized with Ensemble Retriever ONLY")
        logger.info(f"   FAISS weight: {os.getenv('FAISS_WEIGHT', '0.7')}")
        logger.info(f"   BM25 weight: {os.getenv('BM25_WEIGHT', '0.3')}")
        logger.info(f"   FAISS k: {os.getenv('FAISS_RETRIEVAL_K', '50')}")
        logger.info(f"   BM25 k: {os.getenv('BM25_RETRIEVAL_K', '50')}")
        
        stats = self.rag_service.get_stats()
        if stats['faiss']['total_vectors'] > 0:
            logger.info(f"   FAISS: {stats['faiss']['total_vectors']} vectors loaded")
        else:
            logger.warning("   ⚠️ FAISS vector store is empty")
        
        if stats['bm25']['total_chunks'] > 0:
            logger.info(f"   BM25: {stats['bm25']['total_chunks']} chunks loaded")
        else:
            logger.warning("   ⚠️ BM25 retriever is empty")

    async def generate(self, prompt: str, file_ids: Optional[List[str]] = None) -> str:
        """Generate answer using RAG with Ensemble Retriever and optional file filtering"""
        try:
            logger.info(f"📝 User Question: {prompt}")
            if file_ids:
                logger.info(f"   📁 Filtering to file_ids: {file_ids}")
            
            # 1. Retrieve chunks using your custom logic (with reranker)
            retrieved_chunks = await self._retrieve_with_details(prompt, file_ids=file_ids)
            
            if not retrieved_chunks:
                logger.warning("No chunks retrieved from ensemble retriever")
                return "I don't have any relevant information to answer this question. Please add some documents first."
            
            # Log retrieved chunks
            logger.info(f"📚 Retrieved {len(retrieved_chunks)} chunks from {self.retriever_type}:")
            for i, chunk in enumerate(retrieved_chunks, 1):
                chunk_preview = chunk.page_content[:150].replace('\n', ' ')
                logger.info(f"   Chunk {i}: {chunk_preview}...")
                logger.info(f"      Metadata: document_id={chunk.metadata.get('document_id')}, "
                           f"filename={chunk.metadata.get('filename')}")
                if 'reranker_score' in chunk.metadata:
                    logger.info(f"      Reranker Score: {chunk.metadata['reranker_score']:.4f}")
            
            # 2. Build context from retrieved chunks
            context_text = "\n\n---\n\n".join([chunk.page_content for chunk in retrieved_chunks])
            
            # 3. Format the prompt with your context
            formatted_messages = self.prompt.format_messages(context=context_text, input=prompt)
            final_prompt_text = formatted_messages[0].content if formatted_messages else ""
            
            # HIGHLY VISIBLE PROMPT LOGGING
            logger.info("=" * 100)
            logger.info("🤖🤖🤖 FINAL PROMPT SENT TO OLLAMA 🤖🤖🤖")
            logger.info("=" * 100)
            logger.info(f"📊 STATISTICS:")
            logger.info(f"   - Total chunks in context: {len(retrieved_chunks)}")
            logger.info(f"   - Total context length: {len(context_text)} characters")
            logger.info(f"   - User query length: {len(prompt)} characters")
            logger.info("-" * 100)
            logger.info("📝 FULL PROMPT CONTENT:")
            logger.info("-" * 100)
            logger.info(final_prompt_text)
            logger.info("=" * 100)
            
            # 4. ⭐ DIRECTLY CALL THE LLM with your formatted messages ⭐
            # This bypasses the chain and uses YOUR retrieved context
            response = await self.llm.ainvoke(formatted_messages)
            answer = response.content.strip()
            
            logger.info(f"💬 LLM Response: {answer[:300]}...")
            return answer
            
        except Exception as e:
            logger.error(f"LLM RAG generation error: {e}")
            return f"Error generating response: {str(e)}"
    
    async def generate_with_sources(self, prompt: str, file_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate answer and return with source documents"""
        try:
            logger.info(f"📝 User Question: {prompt}")
            if file_ids:
                logger.info(f"   📁 Filtering to file_ids: {file_ids}")
            
            retrieved_chunks = await self._retrieve_with_details(prompt, file_ids=file_ids)
            
            if not retrieved_chunks:
                return {"answer": "I don't have any relevant information.", "sources": [], "retrieval_method": self.retriever_type, "total_chunks_retrieved": 0}
            
            # Build context
            context_text = "\n\n---\n\n".join([chunk.page_content for chunk in retrieved_chunks])
            
            # HIGHLY VISIBLE PROMPT LOGGING WITH SOURCES
            logger.info("=" * 100)
            logger.info("🤖🤖🤖 FINAL PROMPT SENT TO OLLAMA (WITH SOURCES) 🤖🤖🤖")
            logger.info("=" * 100)
            logger.info(f"📊 STATISTICS:")
            logger.info(f"   - Total chunks: {len(retrieved_chunks)}")
            logger.info(f"   - Context length: {len(context_text)} chars")
            logger.info("-" * 100)
            logger.info("📝 FULL PROMPT CONTENT:")
            logger.info("-" * 100)
            logger.info(context_text)
            logger.info("=" * 100)
            
            # ⭐ DIRECTLY CALL THE LLM ⭐
            formatted_messages = self.prompt.format_messages(context=context_text, input=prompt)
            response = await self.llm.ainvoke(formatted_messages)
            answer = response.content.strip()
            
            sources = [{"rank": i, "content_preview": chunk.page_content[:200], "full_content": chunk.page_content, "document_id": chunk.metadata.get('document_id'), "filename": chunk.metadata.get('filename'), "reranker_score": chunk.metadata.get('reranker_score')} for i, chunk in enumerate(retrieved_chunks, 1)]
            
            return {"answer": answer, "sources": sources, "retrieval_method": self.retriever_type, "total_chunks_retrieved": len(retrieved_chunks)}
            
        except Exception as e:
            logger.error(f"LLM RAG generation error: {e}")
            return {"answer": f"Error: {str(e)}", "sources": [], "retrieval_method": self.retriever_type, "error": str(e)}
    
    async def generate_with_ensemble_only(self, prompt: str, k: int = 5, file_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate answer using ONLY ensemble retriever with detailed output"""
        try:
            if file_ids:
                retrieved_chunks = self.rag_service.search_with_file_filtering(prompt, file_ids, k)
            else:
                ensemble_retriever = self.rag_service.get_ensemble_retriever(
                    faiss_weight=float(os.getenv("FAISS_WEIGHT", "0.7")),
                    bm25_weight=float(os.getenv("BM25_WEIGHT", "0.3")),
                    faiss_k=k * 2,
                    bm25_k=k * 2
                )
                retrieved_chunks = await ensemble_retriever.ainvoke(prompt)
                retrieved_chunks = retrieved_chunks[:k]
            
            # Build context
            context_text = "\n\n---\n\n".join([chunk.page_content for chunk in retrieved_chunks])
            
            # HIGHLY VISIBLE PROMPT LOGGING
            logger.info("=" * 100)
            logger.info("🤖🤖🤖 FINAL PROMPT (ENSEMBLE ONLY) SENT TO OLLAMA 🤖🤖🤖")
            logger.info("=" * 100)
            logger.info(f"📊 STATISTICS:")
            logger.info(f"   - Chunks retrieved: {len(retrieved_chunks)}")
            logger.info(f"   - Context length: {len(context_text)} chars")
            logger.info("-" * 100)
            logger.info("📝 PROMPT CONTEXT:")
            logger.info("-" * 100)
            logger.info(context_text)
            logger.info("=" * 100)
            
            # ⭐ DIRECTLY CALL THE LLM ⭐
            formatted_messages = self.prompt.format_messages(context=context_text, input=prompt)
            response = await self.llm.ainvoke(formatted_messages)
            answer = response.content.strip()
            
            return {
                "answer": answer,
                "retrieved_chunks": [{"content": chunk.page_content, "metadata": chunk.metadata} for chunk in retrieved_chunks],
                "weights_used": {"faiss": float(os.getenv("FAISS_WEIGHT", "0.7")), "bm25": float(os.getenv("BM25_WEIGHT", "0.3"))},
                "num_chunks_retrieved": len(retrieved_chunks)
            }
            
        except Exception as e:
            logger.error(f"Ensemble only generation error: {e}")
            return {"answer": f"Error: {str(e)}", "retrieved_chunks": [], "weights_used": None}
    
    async def _retrieve_with_details(self, query: str, k: int = 10, file_ids: Optional[List[str]] = None) -> List[Document]:
        """Retrieve chunks using ensemble retriever with optional file filtering"""
        if file_ids:
            results = self.rag_service.search_with_file_filtering(query, file_ids, k)
            return results
        
        if hasattr(self.rag_service, 'search_with_ensemble'):
            results = self.rag_service.search_with_ensemble(
                query=query,
                final_k=k,
                faiss_weight=float(os.getenv("FAISS_WEIGHT", "0.7")),
                bm25_weight=float(os.getenv("BM25_WEIGHT", "0.3")),
                faiss_k=int(os.getenv("FAISS_RETRIEVAL_K", "50")),
                bm25_k=int(os.getenv("BM25_RETRIEVAL_K", "50"))
            )
            return results
        else:
            if self.ensemble_retriever:
                results = await self.ensemble_retriever.ainvoke(query)
                return results[:k]
            return []
    
    def check_ensemble_status(self) -> Dict[str, Any]:
        """Check if ensemble retriever is properly configured"""
        stats = self.rag_service.get_stats()
        return {
            "ensemble_ready": stats['faiss']['total_vectors'] > 0 and stats['bm25']['total_chunks'] > 0,
            "faiss_ready": stats['faiss']['total_vectors'] > 0,
            "bm25_ready": stats['bm25']['total_chunks'] > 0,
            "faiss_vectors": stats['faiss']['total_vectors'],
            "bm25_chunks": stats['bm25']['total_chunks'],
            "retriever_type": self.retriever_type,
            "config": {"faiss_weight": os.getenv("FAISS_WEIGHT", "0.7"), "bm25_weight": os.getenv("BM25_WEIGHT", "0.3"), "faiss_k": os.getenv("FAISS_RETRIEVAL_K", "50"), "bm25_k": os.getenv("BM25_RETRIEVAL_K", "50")}
        }
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics about the current retrieval setup"""
        stats = self.rag_service.get_stats()
        stats["retriever_type"] = self.retriever_type
        stats["ensemble_active"] = self.ensemble_retriever is not None
        stats["config"] = {
            "faiss_weight": os.getenv("FAISS_WEIGHT", "0.7"),
            "bm25_weight": os.getenv("BM25_WEIGHT", "0.3"),
            "faiss_k": os.getenv("FAISS_RETRIEVAL_K", "50"),
            "bm25_k": os.getenv("BM25_RETRIEVAL_K", "50"),
            "ollama_model": os.getenv("OLLAMA_MODEL", "gemma3:12b"),
            "temperature": os.getenv("OLLAMA_TEMPERATURE", "0.3")
        }
        return stats