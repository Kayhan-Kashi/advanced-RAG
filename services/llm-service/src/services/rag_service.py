# services/rag_service.py
import logging
import os
import pickle
import json
from injector import inject
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from common.kafka.producer import KafkaProducer #type: ignore
from src.core.document_loader import DocumentLoader
from src.core.text_chunker import TextChunker
from src.core.text_preprocessor import TextPreprocessor
from langchain_core.documents import Document
from typing import List, Optional, Dict, Any, Tuple
from src.services.embedding_service import EmbeddingService
from src.core.JinaLangChainWrapper import JinaLangChainWrapper

# Try to import reranker
try:
    from FlagEmbedding import FlagReranker
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("FlagEmbedding not installed. Reranking will be disabled.")

logger = logging.getLogger(__name__)


class RagService:
    """RAG Service with FAISS (dense MMR), BM25 (sparse), and Reranker"""
    
    @inject
    def __init__(self,
                document_loader: DocumentLoader, 
                text_preprocessor: TextPreprocessor, 
                text_chunker: TextChunker,
                embedding_service: EmbeddingService,
                kafka_producer: Optional[KafkaProducer] = None):
        self.doc_loader = document_loader
        self.text_preprocessor = text_preprocessor
        self.text_chunker = text_chunker
        self.embedding_service = embedding_service
        self.kafka_producer = kafka_producer
        self.vector_store = None
        self.bm25_retriever = None
        self.all_chunks = []
        self.reranker = None
        
        # MMR configuration
        self.mmr_fetch_k = int(os.getenv("MMR_FETCH_K", "100"))
        self.mmr_lambda_mult = float(os.getenv("MMR_LAMBDA_MULT", "0.7"))
        
        # Index paths matching Docker volume mounts
        self.faiss_index_path = "/app/faiss_index"  # Mounted from ./models/faiss_index
        self.bm25_index_path = "/app/bm25_index"    # Mounted from ./models/bm25_index
        self.bm25_chunks_file = f"{self.bm25_index_path}/chunks.pkl"
        self.bm25_metadata_file = f"{self.bm25_index_path}/metadata.json"
        
        # For local development fallback (if not running in Docker)
        if not os.path.exists("/app"):
            self.faiss_index_path = "./models/faiss_index"
            self.bm25_index_path = "./models/bm25_index"
            self.bm25_chunks_file = f"{self.bm25_index_path}/chunks.pkl"
            self.bm25_metadata_file = f"{self.bm25_index_path}/metadata.json"
        
        # Create directories if they don't exist
        self._ensure_directories()
        
        # Try to load existing vector stores
        self._load_vector_store()
        self._load_bm25_retriever()
        
        # Try to load reranker
        self._load_reranker()
        
        logger.info("✅ RagService initialized")
        logger.info(f"   FAISS index path: {self.faiss_index_path}")
        logger.info(f"   BM25 index path: {self.bm25_index_path}")
        logger.info(f"   Reranker available: {self.reranker is not None}")
        logger.info(f"   MMR enabled: YES (fetch_k={self.mmr_fetch_k}, lambda={self.mmr_lambda_mult})")
    
    def _ensure_directories(self):
        """Create necessary directories for indexes"""
        os.makedirs(self.faiss_index_path, exist_ok=True)
        os.makedirs(self.bm25_index_path, exist_ok=True)
        logger.info(f"📁 Ensured directories exist: {self.faiss_index_path}, {self.bm25_index_path}")
    
    def _load_reranker(self):
        """Load the reranker model - handles HuggingFace cache structure"""
        if not RERANKER_AVAILABLE:
            logger.warning("FlagEmbedding not available, reranking disabled")
            return
        
        try:
            model_id = 'BAAI/bge-reranker-v2-m3'
            
            if os.path.exists("/app"):
                # Running in Docker
                base_path = '/app/models/BAAI/models--BAAI--bge-reranker-v2-m3'
                
                if os.path.exists(base_path):
                    logger.info(f"✅ Found model directory: {base_path}")
                    
                    # Check if it's in HuggingFace cache format (has snapshots folder)
                    snapshots_path = os.path.join(base_path, 'snapshots')
                    
                    if os.path.exists(snapshots_path):
                        # Find the actual model in snapshots
                        logger.info("Detected HuggingFace cache format, looking in snapshots...")
                        snapshots = [d for d in os.listdir(snapshots_path) if os.path.isdir(os.path.join(snapshots_path, d))]
                        
                        if snapshots:
                            # Use the first/latest snapshot
                            snapshot_hash = snapshots[0]
                            model_path = os.path.join(snapshots_path, snapshot_hash)
                            logger.info(f"✅ Found model snapshot: {model_path}")
                            
                            # Verify config.json exists
                            config_path = os.path.join(model_path, 'config.json')
                            if os.path.exists(config_path):
                                logger.info("Loading model from snapshot...")
                                self.reranker = FlagReranker(
                                    model_path,
                                    use_fp16=True
                                )
                            else:
                                logger.error(f"config.json not found in {model_path}")
                                self.reranker = None
                                return
                        else:
                            logger.error(f"No snapshots found in {snapshots_path}")
                            self.reranker = None
                            return
                    else:
                        # Direct model files (not cache format)
                        logger.info("Loading model directly...")
                        self.reranker = FlagReranker(
                            base_path,
                            use_fp16=True
                        )
                else:
                    logger.error(f"❌ Model directory not found at {base_path}")
                    self.reranker = None
                    return
            else:
                # Running locally
                cache_dir = '../../../models/BAAI'
                model_cache_path = os.path.join(cache_dir, 'models--BAAI--bge-reranker-v2-m3')
                
                if os.path.exists(model_cache_path):
                    logger.info(f"✅ Model found in cache: {model_cache_path}")
                    # Check if it's in snapshots format
                    snapshots_path = os.path.join(model_cache_path, 'snapshots')
                    if os.path.exists(snapshots_path):
                        snapshots = [d for d in os.listdir(snapshots_path) if os.path.isdir(os.path.join(snapshots_path, d))]
                        if snapshots:
                            model_path = os.path.join(snapshots_path, snapshots[0])
                            logger.info(f"Loading from snapshot: {model_path}")
                            self.reranker = FlagReranker(
                                model_path,
                                use_fp16=True
                            )
                        else:
                            self.reranker = FlagReranker(
                                model_id,
                                use_fp16=True,
                                cache_dir=cache_dir
                            )
                    else:
                        self.reranker = FlagReranker(
                            model_cache_path,
                            use_fp16=True
                        )
                else:
                    logger.info(f"📥 Downloading {model_id} to {cache_dir}...")
                    self.reranker = FlagReranker(
                        model_id,
                        use_fp16=True,
                        cache_dir=cache_dir
                    )
            
            logger.info("✅ Reranker loaded successfully!")
            
            # Quick test to verify it works
            test_query = "test"
            test_docs = ["This is a test document."]
            test_scores = self.reranker.compute_score([(test_query, test_docs[0])])
            logger.info(f"   Test reranking: ✓ successful (score: {test_scores[0]:.4f})")
            
        except Exception as e:
            logger.error(f"❌ Failed to load reranker: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            self.reranker = None
    
    def _rerank_chunks(self, query: str, chunks: List[Document], top_k: int = 5) -> List[Document]:
        """
        Rerank chunks using the reranker model and SHOW the difference
        
        Args:
            query: The search query
            chunks: List of Document chunks to rerank
            top_k: Number of top chunks to return after reranking
        
        Returns:
            List of reranked Document chunks
        """
        if not self.reranker or not chunks:
            return chunks[:top_k]
        
        try:
            # Extract texts from chunks
            texts = [chunk.page_content for chunk in chunks]
            
            # Create pairs for reranking
            pairs = [(query, text) for text in texts]
            
            # Calculate relevance scores
            scores = self.reranker.compute_score(pairs)
            
            # Combine chunks with scores
            ranked = list(zip(chunks, scores))
            ranked.sort(key=lambda x: x[1], reverse=True)
            
            # ============ SHOW THE DIFFERENCE ============
            logger.info("=" * 80)
            logger.info("🎯 RERANKER IMPACT ANALYSIS")
            logger.info("=" * 80)
            
            # Show top 5 BEFORE reranking (original order from ensemble)
            logger.info("\n📋 BEFORE RERANKING (Original Ensemble Order):")
            logger.info("-" * 60)
            for i, chunk in enumerate(chunks[:5], 1):
                preview = chunk.page_content[:150].replace('\n', ' ')
                logger.info(f"   Rank {i}: {preview}...")
            
            # Show top 5 AFTER reranking with scores
            logger.info("\n📋 AFTER RERANKING (Re-ordered by Cross-Encoder):")
            logger.info("-" * 60)
            for i, (chunk, score) in enumerate(ranked[:5], 1):
                preview = chunk.page_content[:150].replace('\n', ' ')
                logger.info(f"   Rank {i}: Score={score:.4f} - {preview}...")
            
            # Show position changes
            logger.info("\n📊 POSITION CHANGES:")
            logger.info("-" * 60)
            original_positions = {id(chunk): i for i, chunk in enumerate(chunks)}
            
            for new_pos, (chunk, score) in enumerate(ranked[:5], 1):
                old_pos = original_positions.get(id(chunk), -1) + 1
                if old_pos != new_pos:
                    change = old_pos - new_pos
                    direction = "↑" if change > 0 else "↓"
                    logger.info(f"   Chunk originally at #{old_pos} → #{new_pos} {direction} (score: {score:.4f})")
                else:
                    logger.info(f"   Chunk originally at #{old_pos} → #{new_pos} = (unchanged, score: {score:.4f})")
            
            # Show score distribution
            logger.info("\n📈 SCORE DISTRIBUTION:")
            logger.info("-" * 60)
            high_scores = [s for _, s in ranked if s >= 0.7]
            mid_scores = [s for _, s in ranked if 0.4 <= s < 0.7]
            low_scores = [s for _, s in ranked if s < 0.4]
            logger.info(f"   High relevance (≥0.7): {len(high_scores)} chunks")
            logger.info(f"   Medium relevance (0.4-0.7): {len(mid_scores)} chunks")
            logger.info(f"   Low relevance (<0.4): {len(low_scores)} chunks")
            
            logger.info("=" * 80)
            
            # Store reranker scores on chunks for potential MMR use
            for chunk, score in ranked:
                chunk.metadata['reranker_score'] = score
            
            # Return top_k chunks
            return [chunk for chunk, score in ranked[:top_k]]
            
        except Exception as e:
            logger.error(f"❌ Reranking failed: {e}")
            return chunks[:top_k]
    
    def _load_bm25_retriever(self):
        """Load BM25 retriever from disk if it exists"""
        if os.path.exists(self.bm25_chunks_file):
            try:
                with open(self.bm25_chunks_file, 'rb') as f:
                    self.all_chunks = pickle.load(f)
                
                # Recreate BM25 retriever from saved chunks
                if self.all_chunks:
                    self.bm25_retriever = BM25Retriever.from_documents(self.all_chunks)
                    logger.info(f"✅ Loaded existing BM25 retriever from {self.bm25_index_path}")
                    logger.info(f"   Total chunks: {len(self.all_chunks)}")
                    
                    # Load metadata if exists
                    if os.path.exists(self.bm25_metadata_file):
                        with open(self.bm25_metadata_file, 'r') as f:
                            metadata = json.load(f)
                            logger.info(f"   BM25 metadata: {metadata}")
            except Exception as e:
                logger.warning(f"Failed to load BM25 retriever: {e}")
        else:
            logger.info("No existing BM25 retriever found")
    
    def save_bm25_retriever(self) -> bool:
        """Save BM25 chunks and metadata to disk"""
        if not self.all_chunks:
            logger.warning("No BM25 chunks to save")
            return False
        
        try:
            # Save chunks
            with open(self.bm25_chunks_file, 'wb') as f:
                pickle.dump(self.all_chunks, f)
            
            # Save metadata
            metadata = {
                "total_chunks": len(self.all_chunks),
                "last_updated": str(__import__('datetime').datetime.now()),
                "index_path": self.bm25_index_path,
                "type": "BM25"
            }
            
            with open(self.bm25_metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"💾 Saved BM25 retriever to {self.bm25_index_path}")
            logger.info(f"   Total chunks saved: {len(self.all_chunks)}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save BM25 retriever: {e}")
            return False
    
    def _load_vector_store(self):
        """Load vector store from disk if it exists"""
        index_file = f"{self.faiss_index_path}/index.faiss"
        if os.path.exists(index_file):
            try:
                from src.core.JinaLangChainWrapper import JinaLangChainWrapper
                
                embedding_wrapper = JinaLangChainWrapper(self.embedding_service.model)
                
                self.vector_store = FAISS.load_local(
                    self.faiss_index_path, 
                    embedding_wrapper,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"✅ Loaded existing FAISS vector store from {self.faiss_index_path}")
                logger.info(f"   Total vectors: {self.vector_store.index.ntotal}")
                logger.info(f"   Dimension: {self.vector_store.index.d}")
                
                # Test retrieval to verify it works
                test_query = "test"
                test_results = self.vector_store.similarity_search(test_query, k=1)
                logger.info(f"   Test retrieval: ✓ successful")
                
            except Exception as e:
                logger.warning(f"Failed to load FAISS vector store: {e}")
                import traceback
                logger.debug(traceback.format_exc())
        else:
            logger.info("No existing FAISS vector store found. Will create new one when documents are added.")
    
    def save_vector_store(self) -> bool:
        """Save FAISS vector store to disk"""
        if self.vector_store is None:
            logger.warning("No FAISS vector store to save")
            return False
        
        try:
            self.vector_store.save_local(self.faiss_index_path)
            logger.info(f"💾 Saved FAISS vector store to {self.faiss_index_path}")
            logger.info(f"   Total vectors saved: {self.vector_store.index.ntotal}")
            
            # Save metadata
            metadata = {
                "total_vectors": self.vector_store.index.ntotal,
                "dimension": self.vector_store.index.d,
                "last_updated": str(__import__('datetime').datetime.now()),
                "type": "FAISS",
                "mmr_enabled": True,
                "mmr_fetch_k": self.mmr_fetch_k,
                "mmr_lambda_mult": self.mmr_lambda_mult
            }
            metadata_file = f"{self.faiss_index_path}/metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save FAISS vector store: {e}")
            return False
    
    def store_vector(self, chunks: List[Document]) -> None:
        """Store document chunks in FAISS vector store and BM25"""
        try:
            if not chunks:
                logger.warning("No chunks to store")
                return
            
            # Store chunks for BM25
            self.all_chunks.extend(chunks)
            self._update_bm25_retriever()
            
            # Create wrapper for embeddings
            langchain_compatible_model = JinaLangChainWrapper(self.embedding_service.model)
            
            if self.vector_store is None:
                # Create new vector store
                self.vector_store = FAISS.from_documents(chunks, langchain_compatible_model)
                logger.info(f"✅ Created new FAISS vector store with {len(chunks)} chunks")
            else:
                # Add to existing vector store
                self.vector_store.add_documents(chunks)
                logger.info(f"✅ Added {len(chunks)} chunks to existing FAISS vector store")
            
            logger.info(f"   Total FAISS vectors: {self.vector_store.index.ntotal}")
            logger.info(f"   FAISS dimension: {self.vector_store.index.d}")
            logger.info(f"   Total BM25 chunks: {len(self.all_chunks)}")
            
            # Auto-save after each addition
            self.save_vector_store()
            self.save_bm25_retriever()
            
        except Exception as e:
            logger.error(f"❌ Failed to store vectors: {e}")
            raise
    
    def _update_bm25_retriever(self):
        """Update BM25 retriever with current chunks"""
        if self.all_chunks:
            self.bm25_retriever = BM25Retriever.from_documents(self.all_chunks)
            logger.info(f"🔄 Updated BM25 retriever with {len(self.all_chunks)} total chunks")
    
    def get_faiss_retriever(self, search_kwargs: Optional[dict] = None):
        """
        Get FAISS retriever with MMR ALWAYS enabled for diversity
        
        Args:
            search_kwargs: Search parameters (k, fetch_k, lambda_mult)
                          If not provided, uses configured MMR defaults
        
        Returns:
            FAISS retriever object with MMR or None if vector store is empty
        """
        if self.vector_store is None:
            logger.warning("FAISS vector store is empty, cannot create retriever")
            return None
        
        # Default MMR configuration
        default_kwargs = {
            "k": 5,
            "fetch_k": self.mmr_fetch_k,
            "lambda_mult": self.mmr_lambda_mult
        }
        
        if search_kwargs:
            default_kwargs.update(search_kwargs)
        
        # ALWAYS use MMR search type for diversity
        retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs=default_kwargs
        )
        
        logger.debug(f"   FAISS MMR retriever: k={default_kwargs['k']}, fetch_k={default_kwargs['fetch_k']}, lambda={default_kwargs['lambda_mult']}")
        
        return retriever
    
    def get_bm25_retriever(self, search_kwargs: Optional[dict] = None):
        """
        Get BM25 retriever for use in EnsembleRetriever
        
        Args:
            search_kwargs: Search parameters like k (number of documents)
        
        Returns:
            BM25 retriever object or None if no chunks available
        """
        if self.bm25_retriever is None:
            logger.warning("BM25 retriever is empty, cannot create retriever")
            return None
        
        default_kwargs = {"k": 5}
        if search_kwargs:
            default_kwargs.update(search_kwargs)
        
        # Create new BM25 retriever with specified k value
        if self.all_chunks:
            k = default_kwargs.get("k", 5)
            bm25 = BM25Retriever.from_documents(self.all_chunks)
            bm25.k = k
            return bm25
        
        return self.bm25_retriever
    
    def get_ensemble_retriever(self, 
                              faiss_weight: float = 0.5, 
                              bm25_weight: float = 0.5,
                              faiss_k: int = 50,
                              bm25_k: int = 50,
                              weights: Optional[List[float]] = None) -> Optional[EnsembleRetriever]:
        """
        Create an EnsembleRetriever combining FAISS (dense MMR) and BM25 (sparse) retrievers
        
        Args:
            faiss_weight: Weight for FAISS retriever (if weights not provided)
            bm25_weight: Weight for BM25 retriever (if weights not provided)
            faiss_k: Number of documents to retrieve from FAISS (MMR will use fetch_k internally)
            bm25_k: Number of documents to retrieve from BM25
            weights: Optional list of weights [faiss_weight, bm25_weight]
        
        Returns:
            EnsembleRetriever or None if retrievers are not available
        """
        # Get individual retrievers - FAISS with MMR, BM25 with similarity
        faiss_retriever = self.get_faiss_retriever(search_kwargs={"k": faiss_k})
        bm25_retriever = self.get_bm25_retriever(search_kwargs={"k": bm25_k})
        
        if faiss_retriever is None or bm25_retriever is None:
            logger.error("Cannot create EnsembleRetriever: one or both retrievers are unavailable")
            logger.error(f"  FAISS available: {faiss_retriever is not None}")
            logger.error(f"  BM25 available: {bm25_retriever is not None}")
            return None
        
        # Set weights
        if weights is None:
            weights = [faiss_weight, bm25_weight]
        
        # Create ensemble retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[faiss_retriever, bm25_retriever],
            weights=weights
        )
        
        logger.info(f"🎯 Created EnsembleRetriever with weights: FAISS={weights[0]}, BM25={weights[1]}")
        logger.info(f"   FAISS MMR: k={faiss_k} (fetch_k={self.mmr_fetch_k}, lambda={self.mmr_lambda_mult})")
        logger.info(f"   BM25: k={bm25_k}")
        
        return ensemble_retriever
    
    def search_with_ensemble(self, 
                            query: str, 
                            final_k: int = 5,
                            faiss_weight: float = 0.5,
                            bm25_weight: float = 0.5,
                            faiss_k: int = 50,
                            bm25_k: int = 50,
                            use_reranker: bool = True) -> List[Document]:
        """
        Search using ensemble of FAISS (MMR) and BM25 retrievers, optionally with reranking
        
        Args:
            query: Search query
            final_k: Number of final documents to return
            faiss_weight: Weight for FAISS retriever
            bm25_weight: Weight for BM25 retriever
            faiss_k: Number to retrieve from FAISS (MMR returns diverse results)
            bm25_k: Number to retrieve from BM25 (similarity search)
            use_reranker: Whether to apply reranking after ensemble retrieval
        
        Returns:
            List of documents from ensemble retrieval (reranked if enabled)
        """
        ensemble_retriever = self.get_ensemble_retriever(
            faiss_weight=faiss_weight,
            bm25_weight=bm25_weight,
            faiss_k=faiss_k,
            bm25_k=bm25_k
        )
        
        if ensemble_retriever is None:
            logger.warning("Ensemble retriever not available, falling back to FAISS only")
            return self.search(query, k=final_k)
        
        try:
            # Get results from ensemble (FAISS with MMR + BM25 with similarity)
            results = ensemble_retriever.invoke(query)
            logger.info(f"🔍 Ensemble search (FAISS MMR + BM25) retrieved {len(results)} documents")
            
            # Apply reranking if enabled and available
            if use_reranker and self.reranker is not None:
                logger.info("🎯 Applying cross-encoder reranker to ensemble results")
                results = self._rerank_chunks(query, results, top_k=final_k)
            else:
                # Just limit to final_k
                results = results[:final_k]
            
            logger.info(f"✅ Final results: {len(results)} documents")
            return results
            
        except Exception as e:
            logger.error(f"❌ Ensemble search failed: {e}")
            return []
    
    def search_with_reranker(self, query: str, k: int = 5) -> List[Document]:
        """
        Search using ensemble retriever with MMR and reranking
        
        Args:
            query: Search query
            k: Number of final documents to return
        
        Returns:
            List of reranked documents (diverse from MMR, then reranked for relevance)
        """
        return self.search_with_ensemble(
            query=query,
            final_k=k,
            faiss_weight=float(os.getenv("FAISS_WEIGHT", "0.5")),
            bm25_weight=float(os.getenv("BM25_WEIGHT", "0.5")),
            faiss_k=int(os.getenv("FAISS_RETRIEVAL_K", "50")),
            bm25_k=int(os.getenv("BM25_RETRIEVAL_K", "50")),
            use_reranker=True
        )
    
    # ============ FILE FILTERING METHODS ============
    
    def get_chunks_by_file_ids(self, file_ids: List[str]) -> List[Document]:
        """
        Retrieve chunks that belong to specific file IDs
        
        Args:
            file_ids: List of document/file IDs to filter by
        
        Returns:
            List of Document chunks from the specified files
        """
        if not self.all_chunks:
            logger.warning("No chunks available in storage")
            return []
        
        file_id_set = set(str(fid) for fid in file_ids)
        filtered_chunks = [
            chunk for chunk in self.all_chunks 
            if str(chunk.metadata.get('document_id')) in file_id_set
        ]
        
        logger.info(f"📁 Filtered {len(filtered_chunks)} chunks from {len(file_ids)} files")
        return filtered_chunks
    
    def search_with_file_filtering(self, query: str, file_ids: List[str], k: int = 5, use_reranker: bool = True) -> List[Document]:
        """
        Hybrid search with file filtering:
        1. FAISS: Search all vectors → filter results by file_id (no re-encoding)
        2. BM25: Filter chunks by file_id → then search (fast keyword matching)
        3. Combine both results → deduplicate
        4. Apply reranker (optional)
        
        This avoids re-encoding and gives best of both semantic + keyword search.
        
        Args:
            query: Search query
            file_ids: List of file IDs to restrict search to
            k: Number of results to return
            use_reranker: Whether to apply reranker after retrieval (default: True)
        
        Returns:
            List of relevant documents from the specified files
        """
        if not file_ids:
            logger.warning("No file_ids provided")
            return self.search_with_reranker(query, k)
        
        faiss_results = []
        bm25_results = []
        
        # ============ STEP 1: FAISS - Search all, then filter by file_id ============
        if self.vector_store is not None:
            fetch_k = k * 5  # Get more to ensure enough after filtering
            retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": fetch_k, "fetch_k": self.mmr_fetch_k, "lambda_mult": self.mmr_lambda_mult}
            )
            
            all_results = retriever.invoke(query)
            logger.info(f"🔍 FAISS retrieved {len(all_results)} results from main index")
            
            # Filter FAISS results by file_ids
            file_id_set = set(str(fid) for fid in file_ids)
            faiss_results = [
                doc for doc in all_results 
                if str(doc.metadata.get('document_id')) in file_id_set
            ]
            logger.info(f"📁 FAISS filtered to {len(faiss_results)} chunks from selected files")
        
        # ============ STEP 2: BM25 - Filter by file_id first, then search ============
        if self.bm25_retriever is not None:
            # Get only chunks from selected files
            filtered_chunks = self.get_chunks_by_file_ids(file_ids)
            
            if filtered_chunks:
                # Create BM25 from filtered chunks only
                bm25 = BM25Retriever.from_documents(filtered_chunks)
                bm25.k = k * 2  # Get more from BM25
                bm25_results = bm25.invoke(query)
                logger.info(f"🔍 BM25 retrieved {len(bm25_results)} results from {len(filtered_chunks)} filtered chunks")
            else:
                logger.warning("No chunks found for BM25 filtering")
        
        # ============ STEP 3: Combine and deduplicate ============
        seen_content = set()
        combined_results = []
        
        # Add FAISS results first (semantic quality)
        for doc in faiss_results:
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                combined_results.append(doc)
        
        # Add BM25 results (keyword quality)
        for doc in bm25_results:
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                combined_results.append(doc)
        
        logger.info(f"📊 Combined: {len(combined_results)} unique results (FAISS: {len(faiss_results)}, BM25: {len(bm25_results)})")
        
        # If no results, return empty
        if not combined_results:
            logger.warning("No results found from any retriever")
            return []
        
        # ============ STEP 4: Reranking (optional) ============
        if use_reranker and self.reranker is not None:
            logger.info("🎯 Applying reranker to combined results")
            combined_results = self._rerank_chunks(query, combined_results, top_k=k)
        else:
            combined_results = combined_results[:k]
        
        logger.info(f"✅ Final results: {len(combined_results)} documents")
        return combined_results
    
    # ============ END FILE FILTERING METHODS ============
    
    def compare_retrieval_methods(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Compare retrieval results with and without reranker
        
        Args:
            query: Search query
            k: Number of results to compare
        
        Returns:
            Dictionary with comparison results
        """
        logger.info(f"\n🔬 COMPARING RETRIEVAL METHODS FOR QUERY: {query}")
        logger.info("📍 Note: FAISS always uses MMR for diversity in both methods")
        
        # Without reranker
        logger.info("\n" + "=" * 80)
        logger.info("METHOD 1: Ensemble (FAISS MMR + BM25) ONLY (No Reranker)")
        logger.info("=" * 80)
        results_no_reranker = self.search_with_ensemble(
            query=query,
            final_k=k,
            use_reranker=False
        )
        
        # With reranker
        logger.info("\n" + "=" * 80)
        logger.info("METHOD 2: Ensemble (FAISS MMR + BM25) + Reranker")
        logger.info("=" * 80)
        results_with_reranker = self.search_with_ensemble(
            query=query,
            final_k=k,
            use_reranker=True
        )
        
        # Compare results
        comparison = {
            "query": query,
            "mmr_enabled": True,
            "mmr_fetch_k": self.mmr_fetch_k,
            "mmr_lambda_mult": self.mmr_lambda_mult,
            "without_reranker": [
                {
                    "rank": i+1,
                    "content_preview": doc.page_content[:200],
                    "metadata": doc.metadata
                }
                for i, doc in enumerate(results_no_reranker)
            ],
            "with_reranker": [
                {
                    "rank": i+1,
                    "content_preview": doc.page_content[:200],
                    "metadata": doc.metadata
                }
                for i, doc in enumerate(results_with_reranker)
            ]
        }
        
        # Check if results are different
        different = False
        if len(results_no_reranker) == len(results_with_reranker):
            for i in range(len(results_no_reranker)):
                if results_no_reranker[i].page_content != results_with_reranker[i].page_content:
                    different = True
                    break
        else:
            different = True
        
        comparison["results_changed"] = different
        
        if different:
            logger.info("\n✅ Reranker CHANGED the result order!")
        else:
            logger.info("\n⚠️ Reranker did NOT change the result order for this query")
        
        return comparison
    
    def retrieve_similar_chunks(self, query: str, k: int = 10) -> List[Document]:
        """
        Find the top k most similar chunks for a given query.
        NOTE: This uses similarity search (not MMR) for debugging/comparison purposes.
        For production RAG, use search_with_reranker() instead.
        """
        if self.vector_store is None:
            logger.warning("Vector store is empty, cannot perform retrieval.")
            return []
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"🔍 Retrieved {len(results)} chunks for query: '{query[:30]}...' (similarity search, not MMR)")
            return results
        except Exception as e:
            logger.error(f"❌ Retrieval failed: {e}")
            return []
    
    def prepare_document(self, file_path: str, document_id: str, filename: str, 
                        filetype: str, chunk_size: int = 200, 
                        chunk_overlap: int = 20) -> List[Document]:
        """Load PDF, preprocess, chunk, store, and verify with a retrieval test"""
        logger.info(f"📄 Processing document: {filename}")
        
        try:
            # 1. LOAD
            self.doc_loader.load_pdf(file_path=file_path)
            raw_text = self.doc_loader.text
            logger.info(f"DEBUG [1/3] Raw text length: {len(raw_text) if raw_text else 0}")

            if not raw_text:
                logger.error("❌ No text extracted from document.")
                return []

            # 2. PREPROCESS
            preprocessed_text = self.text_preprocessor.preprocess(raw_text)
            logger.info(f"DEBUG [2/3] Preprocessed text length: {len(preprocessed_text)}")

            if not preprocessed_text:
                logger.error("❌ Preprocessing removed all text.")
                return []

            # 3. CREATE DOCUMENT
            doc = Document(
                page_content=preprocessed_text,
                metadata={
                    "source": file_path,
                    "document_id": document_id,
                    "filename": filename,
                    "filetype": filetype
                }
            )

            # 4. CHUNK
            chunks = self.text_chunker.chunk(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            logger.info(f"DEBUG [3/3] Chunk count: {len(chunks)}")

            if len(chunks) > 0:
                logger.info(f"DEBUG: First chunk preview: {chunks[0].page_content[:120]}")

            # 5. Add metadata
            for chunk in chunks:
                chunk.metadata['document_id'] = document_id
                chunk.metadata['filename'] = filename
                chunk.metadata['filetype'] = filetype

            # 6. Store in FAISS and BM25
            logger.info(f"📊 Storing {len(chunks)} chunks in vector stores...")
            self.store_vector(chunks)

            # 7. Retrieval sanity check (using similarity, not MMR, for baseline)
            if chunks:
                logger.info("🧪 Running retrieval sanity check (Top 10 similarity)...")
                sample_query = "what is the main content"
                top_10 = self.retrieve_similar_chunks(sample_query, k=10)

                logger.info(f"✨ Retrieved {len(top_10)} chunks for verification:")
                for i, doc in enumerate(top_10):
                    logger.info(
                        f"[{i+1}] {doc.page_content[:60]}... "
                        f"(ID: {doc.metadata.get('document_id')})"
                    )

            logger.info(f"✅ Document processing completed: {filename}")
            return chunks

        except Exception as e:
            logger.error(f"❌ Failed to process document: {e}")
            raise
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        """
        Simple similarity search (no ensemble, no reranker).
        For better results, use search_with_reranker() instead.
        """
        if self.vector_store is None:
            logger.warning("No vector store available. Please prepare a document first.")
            return []
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"🔍 Found {len(results)} similar documents (similarity search)")
            return results
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []
    
    def search_with_scores(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Search with similarity scores (not MMR)"""
        if self.vector_store is None:
            logger.warning("No vector store available")
            return []
        
        try:
            results = self.vector_store.similarity_search_with_relevance_scores(query, k=k)
            logger.info(f"🔍 Found {len(results)} results with scores")
            return results
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics for both FAISS and BM25"""
        stats = {
            "faiss": {
                "status": "empty",
                "total_vectors": 0,
                "dimension": 0,
                "index_path": self.faiss_index_path,
                "mmr_enabled": True,
                "mmr_fetch_k": self.mmr_fetch_k,
                "mmr_lambda_mult": self.mmr_lambda_mult
            },
            "bm25": {
                "status": "empty",
                "total_chunks": 0,
                "index_path": self.bm25_index_path
            },
            "reranker": {
                "available": self.reranker is not None,
                "model": "BAAI/bge-reranker-v2-m3" if self.reranker else None
            }
        }
        
        if self.vector_store:
            stats["faiss"] = {
                "status": "active",
                "total_vectors": self.vector_store.index.ntotal,
                "dimension": self.vector_store.index.d,
                "index_path": self.faiss_index_path,
                "mmr_enabled": True,
                "mmr_fetch_k": self.mmr_fetch_k,
                "mmr_lambda_mult": self.mmr_lambda_mult
            }
        
        if self.all_chunks:
            stats["bm25"] = {
                "status": "active",
                "total_chunks": len(self.all_chunks),
                "index_path": self.bm25_index_path
            }
        
        return stats
    
    def clear_vector_store(self) -> None:
        """Clear both FAISS and BM25 vector stores"""
        self.vector_store = None
        self.bm25_retriever = None
        self.all_chunks = []
        logger.info("🗑️ Vector stores cleared")
        
        # Clear FAISS index files
        if os.path.exists(self.faiss_index_path):
            for file in os.listdir(self.faiss_index_path):
                file_path = os.path.join(self.faiss_index_path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            logger.info(f"🗑️ Cleared FAISS index at {self.faiss_index_path}")
        
        # Clear BM25 index files
        if os.path.exists(self.bm25_index_path):
            for file in os.listdir(self.bm25_index_path):
                file_path = os.path.join(self.bm25_index_path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            logger.info(f"🗑️ Cleared BM25 index at {self.bm25_index_path}")