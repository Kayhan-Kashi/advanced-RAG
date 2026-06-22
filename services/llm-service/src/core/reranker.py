# src/core/reranker.py
import logging
import os
import warnings
from typing import Optional, Dict, Any
from injector import inject

logger = logging.getLogger(__name__)

try:
    from FlagEmbedding import FlagReranker
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    logger.warning("FlagEmbedding not installed. Reranking will be disabled.")


class RerankerModel:
    """
    Handles loading and management of the reranker model.
    Auto-downloads from HuggingFace if not found locally.
    """
    
    @inject
    def __init__(self):
        self._reranker = None
        self._is_available = RERANKER_AVAILABLE  # Private attribute
        
        # Model configuration
        self.model_path = os.getenv(
            "RERANKER_MODEL_PATH", 
            "/app/models/BAAI/models--BAAI--bge-reranker-v2-m3"
        )
        self.model_repo_id = os.getenv(
            "RERANKER_REPO_ID",
            "BAAI/bge-reranker-v2-m3"
        )
        
        # Model settings
        self.use_fp16 = os.getenv("RERANKER_USE_FP16", "true").lower() == "true"
        
        # Optimization settings
        self.batch_size = int(os.getenv("RERANKER_BATCH_SIZE", "32"))
        self.max_length = int(os.getenv("RERANKER_MAX_LENGTH", "512"))
        self.limit = int(os.getenv("RERANKER_LIMIT", "30"))
        self.skip_scores = os.getenv("RERANKER_SKIP_SCORES", "false").lower() == "true"
        
        # Cache
        self._score_cache = {}
        self._cache_max_size = int(os.getenv("RERANKER_CACHE_SIZE", "1000"))
        
        # Load the model
        self._load()
        
        logger.info("✅ RerankerModel initialized")
        logger.info(f"   Available: {self.is_available()}")
        logger.info(f"   FP16: {self.use_fp16}")
        logger.info(f"   Batch size: {self.batch_size}")
        logger.info(f"   Max length: {self.max_length}")
        logger.info(f"   Limit: {self.limit}")
        logger.info(f"   Cache size: {self._cache_max_size}")
    
    def is_available(self) -> bool:
        """Check if reranker is available and loaded."""
        return self._reranker is not None and self._is_available
    
    @property
    def reranker(self):
        """Get the reranker instance."""
        return self._reranker
    
    def _load(self):
        """
        Load reranker model with auto-download if not found.
        """
        if not self._is_available:
            logger.warning("⚠️ FlagEmbedding not available. Reranking disabled.")
            return
        
        logger.info("=" * 70)
        logger.info("🔄 LOADING RERANKER MODEL")
        logger.info("=" * 70)
        
        try:
            # Suppress annoying warnings
            logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
            warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
            
            base_path = self.model_path
            logger.info(f"📂 Checking path: {base_path}")
            
            # ================================================================
            # CHECK IF MODEL EXISTS, DOWNLOAD IF NOT
            # ================================================================
            config_path = os.path.join(base_path, "config.json")
            
            if not os.path.exists(config_path):
                logger.info(f"❌ Reranker model not found at: {base_path}")
                logger.info(f"📥 Downloading from HuggingFace...")
                logger.info(f"   Repository: {self.model_repo_id}")
                
                os.makedirs(base_path, exist_ok=True)
                
                from huggingface_hub import snapshot_download
                
                snapshot_download(
                    repo_id=self.model_repo_id,
                    local_dir=base_path,
                    ignore_patterns=[
                        "*.msgpack",      # Flax
                        "flax_model*",    # Flax
                        "tf_model*",      # TensorFlow
                        "*.onnx",         # ONNX
                        "onnx/*"          # ONNX
                    ]
                )
                
                logger.info(f"✅ Reranker model downloaded to: {base_path}")
            
            logger.info(f"✅ Model directory found: {base_path}")
            
            # Check for snapshots (HuggingFace cache format)
            snapshots_path = os.path.join(base_path, 'snapshots')
            model_path = base_path
            
            if os.path.exists(snapshots_path):
                logger.info(f"📁 Found snapshots directory: {snapshots_path}")
                snapshots = [d for d in os.listdir(snapshots_path) 
                           if os.path.isdir(os.path.join(snapshots_path, d))]
                if snapshots:
                    model_path = os.path.join(snapshots_path, snapshots[0])
                    logger.info(f"   Using snapshot: {snapshots[0]}")
                    logger.info(f"   Model path: {model_path}")
            else:
                logger.info(f"📁 Using model path: {model_path}")
            
            # Verify config.json exists
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                logger.info(f"✅ config.json found")
            else:
                logger.warning(f"⚠️ config.json not found at {config_path}")
            
            # Log what we're loading
            logger.info("-" * 70)
            logger.info("📦 Loading FlagReranker with settings:")
            logger.info(f"   Model path: {model_path}")
            logger.info(f"   FP16: {self.use_fp16}")
            logger.info(f"   Device: {'cuda' if self.use_fp16 else 'cpu'}")
            logger.info("-" * 70)
            logger.info("⏳ Loading model into memory (this may take a moment)...")
            
            # Load the model
            self._reranker = FlagReranker(
                model_path, 
                use_fp16=self.use_fp16,
                device="cuda" if self.use_fp16 else "cpu",
            )
            
            logger.info("=" * 70)
            logger.info(f"✅ Reranker loaded successfully!")
            logger.info(f"   Path: {model_path}")
            logger.info(f"   FP16: {self.use_fp16}")
            logger.info(f"   Device: {'cuda' if self.use_fp16 else 'cpu'}")
            logger.info("=" * 70)
                    
        except Exception as e:
            logger.error("=" * 70)
            logger.error(f"❌ Failed to load reranker: {e}")
            logger.error("   Falling back to no reranker. Retrieval will still work.")
            logger.error("=" * 70)
            self._reranker = None
    
    def compute_scores(self, query: str, texts: list) -> list:
        """
        Compute reranker scores for a query and list of texts.
        
        Args:
            query: The query string
            texts: List of text strings to rerank
        
        Returns:
            List of scores
        """
        if not self._reranker or not texts:
            return []
        
        try:
            batch_pairs = [(query, text) for text in texts]
            scores = self._reranker.compute_score(batch_pairs)
            
            # Handle different return types
            if isinstance(scores, float):
                scores = [scores]
            elif not isinstance(scores, list):
                scores = list(scores)
            
            return scores
            
        except Exception as e:
            logger.error(f"Reranker scoring failed: {e}")
            return [0.0] * len(texts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reranker statistics."""
        return {
            "available": self.is_available(),
            "model_path": self.model_path,
            "model_repo_id": self.model_repo_id,
            "use_fp16": self.use_fp16,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "limit": self.limit,
            "cache_size": len(self._score_cache),
            "cache_max_size": self._cache_max_size,
            "skip_scores": self.skip_scores
        }
    
    def clear_cache(self):
        """Clear the score cache."""
        self._score_cache.clear()
        logger.info("🗑️ Reranker cache cleared")