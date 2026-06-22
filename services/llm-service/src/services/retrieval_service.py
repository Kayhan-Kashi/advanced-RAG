# services/retrieval_service.py
import logging
import os
from typing import List, Optional, Dict, Any
from injector import inject
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from src.services.ingestion_service import IngestionService
from src.core.reranker import RerankerModel

logger = logging.getLogger(__name__)


class RetrievalService:
    """
    Handles retrieval operations using indexes from IngestionService.
    
    Responsibilities:
    - FAISS similarity search (with MMR)
    - BM25 keyword search
    - Ensemble combination
    - Cross-encoder reranking (optimized for speed)
    - File filtering
    """
    
    @inject
    def __init__(
        self, 
        ingestion_service: IngestionService,
        reranker_model: RerankerModel
    ):
        self.ingestion = ingestion_service
        self.reranker_model = reranker_model
        
        # MMR configuration
        self.mmr_fetch_k = int(os.getenv("MMR_FETCH_K", "200"))
        self.mmr_lambda_mult = float(os.getenv("MMR_LAMBDA_MULT", "0.5"))
        
        # Ensemble weights
        self.default_faiss_weight = float(os.getenv("FAISS_WEIGHT", "0.6"))
        self.default_bm25_weight = float(os.getenv("BM25_WEIGHT", "0.4"))
        
        # Retrieval counts
        self.default_faiss_k = int(os.getenv("FAISS_RETRIEVAL_K", "100"))
        self.default_bm25_k = int(os.getenv("BM25_RETRIEVAL_K", "100"))
        
        # Reranking configuration
        self.rerank_top_k = int(os.getenv("RERANK_TOP_K", "10"))
        self.enable_reranker = os.getenv("ENABLE_RERANKER", "true").lower() == "true"
        
        # Score cache (using reranker_model's cache)
        self._score_cache = self.reranker_model._score_cache
        self._cache_max_size = self.reranker_model._cache_max_size
        
        logger.info("✅ RetrievalService initialized")
        logger.info(f"   Reranker available: {self.reranker_model.is_available()}")
        logger.info(f"   Reranker batch size: {self.reranker_model.batch_size}")
        logger.info(f"   Reranker FP16: {self.reranker_model.use_fp16}")
        logger.info(f"   Reranker max length: {self.reranker_model.max_length}")
        logger.info(f"   MMR: fetch_k={self.mmr_fetch_k}, lambda={self.mmr_lambda_mult}")
        logger.info(f"   Ensemble: FAISS={self.default_faiss_weight}, BM25={self.default_bm25_weight}")
        logger.info(f"   Retrieval k: FAISS={self.default_faiss_k}, BM25={self.default_bm25_k}")
    
    def retrieve(
        self, 
        query: str, 
        k: int = 5, 
        file_ids: Optional[List[str]] = None,
        use_reranker: Optional[bool] = None
    ) -> List[Document]:
        """
        Main retrieval pipeline with optimized parameters:
        1. FAISS (with MMR) + BM25 ensemble
        2. Optional file filtering
        3. Reranking (cross-encoder) - optimized for speed
        """
        # Check if we should use reranker
        if use_reranker is None:
            use_reranker = self.enable_reranker and self.reranker_model.is_available()
        
        # Get more candidates for reranking
        candidate_k = k * 3 if use_reranker else k
        if file_ids:
            candidate_k = candidate_k * 2
        
        # Step 1: Get results from FAISS and BM25
        faiss_results = self._faiss_search(query, candidate_k)
        bm25_results = self._bm25_search(query, candidate_k)
        
        # Step 2: Filter by file_ids if provided
        if file_ids:
            file_id_set = set(str(fid) for fid in file_ids)
            faiss_results = [d for d in faiss_results if str(d.metadata.get('document_id')) in file_id_set]
            bm25_results = [d for d in bm25_results if str(d.metadata.get('document_id')) in file_id_set]
            logger.info(f"📁 Filtered to {len(faiss_results)} FAISS, {len(bm25_results)} BM25 results")
        
        # Step 3: Combine and deduplicate
        combined = self._combine_results(faiss_results, bm25_results)
        logger.info(f"📊 Combined {len(combined)} unique results")
        
        if not combined:
            return []
        
        # Step 4: Rerank if enabled
        if use_reranker and self.reranker_model.is_available():
            combined = self._rerank_optimized(query, combined, top_k=k)
        else:
            combined = combined[:k]
        
        logger.info(f"✅ Returning {len(combined)} results")
        return combined
    
    def _faiss_search(self, query: str, k: int) -> List[Document]:
        """FAISS search with MMR"""
        vector_store = self.ingestion.get_vector_store()
        if vector_store is None:
            return []
        
        fetch_k = min(k * 2, self.mmr_fetch_k)
        
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": fetch_k,
                "lambda_mult": self.mmr_lambda_mult
            }
        )
        results = retriever.invoke(query)
        logger.debug(f"FAISS: {len(results)} results (k={k}, fetch_k={fetch_k})")
        return results
    
    def _bm25_search(self, query: str, k: int) -> List[Document]:
        """BM25 keyword search"""
        all_chunks = self.ingestion.get_all_chunks()
        if not all_chunks:
            return []
        
        bm25 = BM25Retriever.from_documents(all_chunks)
        bm25.k = k
        results = bm25.invoke(query)
        logger.debug(f"BM25: {len(results)} results (k={k})")
        return results
    
    def _combine_results(self, faiss_results: List[Document], bm25_results: List[Document]) -> List[Document]:
        """Combine and deduplicate results from both retrievers"""
        seen = set()
        combined = []
        
        for doc in faiss_results:
            key = doc.page_content[:200]
            if key not in seen:
                seen.add(key)
                combined.append(doc)
        
        for doc in bm25_results:
            key = doc.page_content[:200]
            if key not in seen:
                seen.add(key)
                combined.append(doc)
        
        return combined
    
    def _rerank_optimized(self, query: str, chunks: List[Document], top_k: int) -> List[Document]:
        """
        Optimized reranking using cross-encoder with:
        - Batch processing
        - Text truncation
        - Score caching
        - Parallel processing (via batch)
        """
        if not self.reranker_model.is_available() or not chunks:
            return chunks[:top_k]
        
        try:
            # Limit chunks to rerank
            rerank_limit = min(len(chunks), self.reranker_model.limit)
            chunks_to_rerank = chunks[:rerank_limit]
            
            # Truncate text for speed
            texts = []
            for chunk in chunks_to_rerank:
                content = chunk.page_content
                if len(content) > self.reranker_model.max_length:
                    content = content[:self.reranker_model.max_length]
                texts.append(content)
            
            # Check cache for scores
            cached_scores = []
            chunks_to_compute = []
            chunks_to_compute_indices = []
            
            for i, text in enumerate(texts):
                cache_key = f"{query[:100]}:{text[:100]}"
                if cache_key in self._score_cache:
                    cached_scores.append(self._score_cache[cache_key])
                else:
                    chunks_to_compute.append(text)
                    chunks_to_compute_indices.append(i)
                    cached_scores.append(None)
            
            # Compute scores for uncached chunks
            if chunks_to_compute:
                # Process in batches
                batch_size = self.reranker_model.batch_size
                computed_scores = []
                
                for i in range(0, len(chunks_to_compute), batch_size):
                    batch_texts = chunks_to_compute[i:i + batch_size]
                    batch_scores = self.reranker_model.compute_scores(query, batch_texts)
                    computed_scores.extend(batch_scores)
                
                # Update cache
                for idx, score in zip(chunks_to_compute_indices, computed_scores):
                    cache_key = f"{query[:100]}:{texts[idx][:100]}"
                    self._score_cache[cache_key] = score
                    
                    # Limit cache size
                    if len(self._score_cache) > self._cache_max_size:
                        self._score_cache.pop(next(iter(self._score_cache)))
                    
                    cached_scores[idx] = score
            
            # Pair chunks with scores and sort
            ranked = list(zip(chunks_to_rerank, cached_scores))
            ranked.sort(key=lambda x: x[1], reverse=True)
            
            # Store scores on chunks
            if not self.reranker_model.skip_scores:
                for chunk, score in ranked:
                    chunk.metadata['reranker_score'] = score
            
            # Log results (top 3 only for speed)
            logger.info(f"🎯 Reranked {len(ranked)} chunks:")
            for i, (chunk, score) in enumerate(ranked[:3], 1):
                preview = chunk.page_content[:100].replace('\n', ' ')
                filename = chunk.metadata.get('filename', 'unknown')
                logger.info(f"   {i}. Score={score:.4f} - {preview}... (from: {filename})")
            
            return [chunk for chunk, _ in ranked[:top_k]]
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return chunks[:top_k]
    
    def search_with_ensemble(
        self,
        query: str,
        k: int = 5,
        faiss_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
        faiss_k: Optional[int] = None,
        bm25_k: Optional[int] = None
    ) -> List[Document]:
        """Search using ensemble retriever with custom weights."""
        faiss_weight = faiss_weight if faiss_weight is not None else self.default_faiss_weight
        bm25_weight = bm25_weight if bm25_weight is not None else self.default_bm25_weight
        faiss_k = faiss_k if faiss_k is not None else self.default_faiss_k
        bm25_k = bm25_k if bm25_k is not None else self.default_bm25_k
        
        faiss_retriever = self._get_faiss_retriever(k=faiss_k)
        bm25_retriever = self._get_bm25_retriever(k=bm25_k)
        
        if faiss_retriever is None or bm25_retriever is None:
            logger.warning("Ensemble retriever not available")
            return self._faiss_search(query, k) or self._bm25_search(query, k)
        
        ensemble = EnsembleRetriever(
            retrievers=[faiss_retriever, bm25_retriever],
            weights=[faiss_weight, bm25_weight]
        )
        
        results = ensemble.invoke(query)
        return results[:k]
    
    def _get_faiss_retriever(self, k: int = 100):
        """Get FAISS retriever with optimized MMR"""
        vector_store = self.ingestion.get_vector_store()
        if vector_store is None:
            return None
        
        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": self.mmr_fetch_k,
                "lambda_mult": self.mmr_lambda_mult
            }
        )
    
    def _get_bm25_retriever(self, k: int = 100):
        """Get BM25 retriever"""
        all_chunks = self.ingestion.get_all_chunks()
        if not all_chunks:
            return None
        
        bm25 = BM25Retriever.from_documents(all_chunks)
        bm25.k = k
        return bm25
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics"""
        return {
            "reranker": self.reranker_model.get_stats(),
            "mmr": {
                "fetch_k": self.mmr_fetch_k,
                "lambda_mult": self.mmr_lambda_mult
            },
            "ensemble": {
                "faiss_weight": self.default_faiss_weight,
                "bm25_weight": self.default_bm25_weight,
                "faiss_k": self.default_faiss_k,
                "bm25_k": self.default_bm25_k
            },
            "rerank": {
                "top_k": self.rerank_top_k,
                "enabled": self.enable_reranker
            }
        }
    
    def clear_cache(self):
        """Clear reranker score cache"""
        self.reranker_model.clear_cache()