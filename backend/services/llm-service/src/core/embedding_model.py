import os
import logging
from injector import inject
from transformers import AutoModel
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

class EmbeddingModel:
    @inject
    def __init__(self):
        self.model_path = os.getenv("MODEL_PATH", "/app/models/snapshot/jina-embeddings-v3")
        self.repo_id = os.getenv("MODEL_REPO_ID", "jinaai/jina-embeddings-v3")
        
        # If TRANSFORMERS_OFFLINE=1, local_files_only=True
        self.is_offline = os.getenv("TRANSFORMERS_OFFLINE", "0") == "1"
        
        # Check if model exists, download if not
        if not os.path.exists(self.model_path) or not os.path.exists(os.path.join(self.model_path, "config.json")):
            if self.is_offline:
                raise FileNotFoundError(
                    f"❌ Model not found at {self.model_path} and offline mode is enabled.\n"
                    f"   Please disable offline mode or download the model first."
                )
            logger.info(f"📥 Model not found at {self.model_path}. Downloading from HuggingFace...")
            logger.info(f"   Repository: {self.repo_id}")
            os.makedirs(self.model_path, exist_ok=True)
            snapshot_download(
                repo_id=self.repo_id,
                local_dir=self.model_path,
                ignore_patterns=[
                    "*.msgpack",      
                    "flax_model*",    
                    "tf_model*",     
                    "*.onnx",         
                    "onnx/*"          
                ]
            )
            logger.info(f"✅ Model downloaded to {self.model_path}")
        
        self._validate_model()
        
        self._model = self._load_model()
        
        logger.info(f"✅ EmbeddingModel initialized")
        logger.info(f"   Model: {self.repo_id}")
        logger.info(f"   Path: {self.model_path}")
        logger.info(f"   Offline mode: {self.is_offline}")

    def _validate_model(self):
        """Ensures the directory and config exist before attempting load."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"❌ Model path does not exist: {self.model_path}")
            
        config_path = os.path.join(self.model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"❌ 'config.json' not found in {self.model_path}. Is the directory empty?")

    def _load_model(self):
        """Loads the model from disk."""
        logger.info(f"📦 Loading embedding model from {self.model_path}...")
        logger.info(f"   Repository: {self.repo_id}")
        logger.info(f"   Offline mode: {self.is_offline}")
        try:
            model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=self.is_offline
            )
            logger.info("✅ Model loaded successfully!")
            return model
        except Exception as e:
            logger.error(f"❌ Failed to load model: {str(e)}")
            raise e

    @property
    def model(self):
        return self._model

    def embed_texts(self, texts: list[str], task: str = "text-matching"):
        """Embed texts using the loaded model."""
        return self._model.encode(texts, task=task)