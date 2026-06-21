import logging
import os
import pickle
import json
from typing import Any, Dict, List, Optional
from injector import inject
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src.core.document_loader import DocumentLoader
from src.core.text_chunker import TextChunker
from src.core.text_preprocessor import TextPreprocessor
from src.core.jina_wrapper import JinaLangChainWrapper
from src.core.embedding_model import EmbeddingModel

logger = logging.getLogger(__name__)


class IngestionService:
    """
    Handles document ingestion: loading, preprocessing, chunking, and indexing.
    Responsible for creating and updating FAISS and BM25 indexes.
    Preserves page metadata from documents.
    """
    
    @inject
    def __init__(
        self,
        document_loader: DocumentLoader,
        text_preprocessor: TextPreprocessor,
        text_chunker: TextChunker,
        embedding_model: EmbeddingModel
    ):
        self.doc_loader = document_loader
        self.text_preprocessor = text_preprocessor
        self.text_chunker = text_chunker
        self.embedding_model = embedding_model
        
        if os.path.exists("/app"):
            base_dir = "/app"
        else:
            base_dir = "."
        
        self.faiss_index_path = os.getenv("FAISS_INDEX_PATH", os.path.join(base_dir, "faiss_index"))
        self.bm25_index_path = os.getenv("BM25_INDEX_PATH", os.path.join(base_dir, "bm25_index"))
        
        if not self.faiss_index_path:
            self.faiss_index_path = os.path.join(base_dir, "faiss_index")
        if not self.bm25_index_path:
            self.bm25_index_path = os.path.join(base_dir, "bm25_index")
        
        self.bm25_chunks_file = os.path.join(self.bm25_index_path, "chunks.pkl")
        self.bm25_metadata_file = os.path.join(self.bm25_index_path, "metadata.json")
        
        self.vector_store = None
        self.bm25_retriever = None
        self.all_chunks = []
        
        self.mmr_fetch_k = int(os.getenv("MMR_FETCH_K", "200"))
        self.mmr_lambda_mult = float(os.getenv("MMR_LAMBDA_MULT", "0.5"))
        
        self.default_chunk_size = int(os.getenv("DEFAULT_CHUNK_SIZE", "500"))
        self.default_chunk_overlap = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "50"))
        
        self.faiss_index_type = os.getenv("FAISS_INDEX_TYPE", "flat")
        
        self._ensure_directories()
        self._load_vector_store()
        self._load_bm25_retriever()
        
        logger.info("✅ IngestionService initialized")
        logger.info(f"   FAISS index: {self.faiss_index_path}")
        logger.info(f"   BM25 index: {self.bm25_index_path}")
        logger.info(f"   FAISS vectors: {self.vector_store.index.ntotal if self.vector_store else 0}")
        logger.info(f"   BM25 chunks: {len(self.all_chunks)}")
        logger.info(f"   MMR: fetch_k={self.mmr_fetch_k}, lambda={self.mmr_lambda_mult}")
        logger.info(f"   Chunking: size={self.default_chunk_size}, overlap={self.default_chunk_overlap}")
    
    def _ensure_directories(self):
        """Create necessary directories for indexes"""
        os.makedirs(self.faiss_index_path, exist_ok=True)
        os.makedirs(self.bm25_index_path, exist_ok=True)
        logger.debug(f"📁 Ensured directories: {self.faiss_index_path}, {self.bm25_index_path}")
    
    def _load_vector_store(self):
        """Load FAISS vector store from disk"""
        index_file = os.path.join(self.faiss_index_path, "index.faiss")
        if os.path.exists(index_file):
            try:
                embedding_wrapper = JinaLangChainWrapper(self.embedding_model.model)
                self.vector_store = FAISS.load_local(
                    self.faiss_index_path,
                    embedding_wrapper,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"✅ Loaded FAISS vector store: {self.vector_store.index.ntotal} vectors")
            except Exception as e:
                logger.warning(f"Failed to load FAISS vector store: {e}")
    
    def _load_bm25_retriever(self):
        """Load BM25 retriever from disk"""
        if os.path.exists(self.bm25_chunks_file):
            try:
                with open(self.bm25_chunks_file, 'rb') as f:
                    self.all_chunks = pickle.load(f)
                if self.all_chunks:
                    self.bm25_retriever = BM25Retriever.from_documents(self.all_chunks)
                    logger.info(f"✅ Loaded BM25 retriever: {len(self.all_chunks)} chunks")
            except Exception as e:
                logger.warning(f"Failed to load BM25 retriever: {e}")
    
    def save_vector_store(self) -> bool:
        """Save FAISS vector store to disk"""
        if self.vector_store is None:
            logger.warning("No FAISS vector store to save")
            return False
        try:
            self.vector_store.save_local(self.faiss_index_path)
            logger.info(f"💾 Saved FAISS vector store: {self.vector_store.index.ntotal} vectors")
            return True
        except Exception as e:
            logger.error(f"Failed to save FAISS: {e}")
            return False
    
    def save_bm25_retriever(self) -> bool:
        """Save BM25 chunks to disk"""
        if not self.all_chunks:
            logger.warning("No BM25 chunks to save")
            return False
        try:
            with open(self.bm25_chunks_file, 'wb') as f:
                pickle.dump(self.all_chunks, f)
            
            metadata = {
                "total_chunks": len(self.all_chunks),
                "last_updated": str(__import__('datetime').datetime.now()),
                "index_path": self.bm25_index_path,
                "type": "BM25",
                "chunk_size": self.default_chunk_size,
                "chunk_overlap": self.default_chunk_overlap,
                "has_page_metadata": True
            }
            with open(self.bm25_metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"💾 Saved BM25 retriever: {len(self.all_chunks)} chunks")
            return True
        except Exception as e:
            logger.error(f"Failed to save BM25: {e}")
            return False
    
    def store_chunks(self, chunks: List[Document]) -> None:
        """
        Store chunks in both FAISS and BM25 indexes.
        
        Args:
            chunks: List of Document chunks to index
        """
        if not chunks:
            logger.warning("No chunks to store")
            return
        
        # Update BM25
        self.all_chunks.extend(chunks)
        self.bm25_retriever = BM25Retriever.from_documents(self.all_chunks)
        logger.info(f"   BM25: {len(self.all_chunks)} total chunks")
        
        # Update FAISS
        langchain_compatible_model = JinaLangChainWrapper(self.embedding_model.model)
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(chunks, langchain_compatible_model)
            logger.info(f"✅ Created FAISS store: {len(chunks)} chunks")
        else:
            self.vector_store.add_documents(chunks)
            logger.info(f"✅ Added {len(chunks)} chunks to FAISS (total: {self.vector_store.index.ntotal})")
        
        self.save_vector_store()
        self.save_bm25_retriever()
    
    def ingest_document(
        self,
        file_path: str,
        document_id: str,
        filename: str,
        filetype: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        chunk_by_page: bool = True
    ) -> List[Document]:
        """
        Load, preprocess, chunk, and index a document with PAGE METADATA preserved.
        
        Args:
            file_path: Path to the document file
            document_id: Unique identifier for the document
            filename: Display name of the document
            filetype: Type of document (pdf, docx, etc.)
            chunk_size: Size of each chunk in characters (defaults to 500)
            chunk_overlap: Overlap between chunks (defaults to 50)
            chunk_by_page: If True, chunk each page separately (preserves page boundaries)
        
        Returns:
            List of created chunks with page metadata
        """
        if chunk_size is None:
            chunk_size = self.default_chunk_size
        if chunk_overlap is None:
            chunk_overlap = self.default_chunk_overlap
        
        logger.info(f"📄 Processing document: {filename}")
        logger.info(f"   Chunk size: {chunk_size}, overlap: {chunk_overlap}")
        logger.info(f"   Chunk by page: {chunk_by_page}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.doc_loader.load_pdf(file_path=file_path)
        
        pages = self.doc_loader.get_pages()
        total_pages = self.doc_loader.get_page_count()
        page_documents = self.doc_loader.get_page_documents()
        
        if not page_documents:
            raise ValueError("No text extracted from document")
        
        logger.info(f"   Loaded: {total_pages} pages")
        
        all_chunks = []
        
        if chunk_by_page:
            all_chunks = self._process_pages_separately(
                page_documents, document_id, filename, filetype, 
                chunk_size, chunk_overlap, total_pages
            )
        else:
            all_chunks = self._process_pages_combined(
                page_documents, document_id, filename, filetype,
                chunk_size, chunk_overlap, total_pages
            )
        
        if all_chunks:
            self.store_chunks(all_chunks)
            logger.info(f"✅ Document processed: {filename}")
            logger.info(f"   Total chunks: {len(all_chunks)}")
            
            sample = all_chunks[0]
            logger.info(f"   Sample: Page {sample.metadata.get('page_number')}/{sample.metadata.get('total_pages')} "
                       f"- {sample.metadata.get('filename')}")
        
        return all_chunks
    
    def _process_pages_separately(
        self,
        page_documents: List[Document],
        document_id: str,
        filename: str,
        filetype: str,
        chunk_size: int,
        chunk_overlap: int,
        total_pages: int
    ) -> List[Document]:
        """
        Process each page separately to preserve page boundaries.
        Each page's chunks will have accurate page_number metadata.
        """
        all_chunks = []
        
        for page_doc in page_documents:
            page_num = page_doc.metadata.get('page_number', 1)
            page_text = page_doc.page_content
            
            preprocessed = self.text_preprocessor.preprocess(page_text)
            if not preprocessed:
                logger.warning(f"   Skipping page {page_num} - no text after preprocessing")
                continue
            
            doc = Document(
                page_content=preprocessed,
                metadata={
                    "source": page_doc.metadata.get('source', ''),
                    "document_id": document_id,
                    "filename": filename,
                    "filetype": filetype,
                    "page_number": page_num,  
                    "total_pages": total_pages,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "page_char_count": len(preprocessed),
                    "page_word_count": len(preprocessed.split()),
                    "has_page_metadata": True
                }
            )
            
            page_chunks = self.text_chunker.chunk(
                doc,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            for i, chunk in enumerate(page_chunks):
                chunk.metadata.update({
                    'document_id': document_id,
                    'filename': filename,
                    'filetype': filetype,
                    'page_number': page_num,  
                    'total_pages': total_pages,  
                    'chunk_index_in_page': i,
                    'total_chunks_in_page': len(page_chunks),
                    'has_page_metadata': True
                })
            
            all_chunks.extend(page_chunks)
        
        for i, chunk in enumerate(all_chunks):
            chunk.metadata['chunk_index'] = i
            chunk.metadata['total_chunks'] = len(all_chunks)
        
        logger.info(f"   Processed {total_pages} pages -> {len(all_chunks)} chunks")
        
        return all_chunks
    
    def _process_pages_combined(
        self,
        page_documents: List[Document],
        document_id: str,
        filename: str,
        filetype: str,
        chunk_size: int,
        chunk_overlap: int,
        total_pages: int
    ) -> List[Document]:
        """
        Combine all pages and chunk together.
        Page metadata is approximated based on position.
        """
        combined_text = ""
        page_markers = [] 
        
        for page_doc in page_documents:
            page_num = page_doc.metadata.get('page_number', 1)
            page_text = page_doc.page_content
            
            preprocessed = self.text_preprocessor.preprocess(page_text)
            if not preprocessed:
                continue
            
            start_pos = len(combined_text)
            combined_text += preprocessed
            end_pos = len(combined_text)
            page_markers.append((page_num, start_pos, end_pos))
        
        if not combined_text:
            raise ValueError("No text after preprocessing")
        
        combined_doc = Document(
            page_content=combined_text,
            metadata={
                "document_id": document_id,
                "filename": filename,
                "filetype": filetype,
                "total_pages": total_pages,
                "page_numbers": [p[0] for p in page_markers],
                "has_page_metadata": True
            }
        )
        
        chunks = self.text_chunker.chunk(
            combined_doc,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        for i, chunk in enumerate(chunks):
            chunk_start = chunk.metadata.get('start_index', 0)
            chunk_end = chunk_start + len(chunk.page_content)
            
            page_num = 1
            pages_covered = []
            
            for p_num, start_pos, end_pos in page_markers:
                if chunk_start <= end_pos and chunk_end >= start_pos:
                    pages_covered.append(p_num)
            
            if pages_covered:
                page_num = pages_covered[0]
            
            chunk.metadata.update({
                'document_id': document_id,
                'filename': filename,
                'filetype': filetype,
                'page_number': page_num,  
                'pages_covered': pages_covered, 
                'total_pages': total_pages,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap,
                'has_page_metadata': True
            })
        
        logger.info(f"   Combined {total_pages} pages -> {len(chunks)} chunks")
        
        return chunks
    
    
    def get_chunks_by_page(self, document_id: str, page_number: int) -> List[Document]:
        """
        Get all chunks for a specific page of a document.
        
        Args:
            document_id: Document ID
            page_number: Page number to filter by
        
        Returns:
            List of chunks from the specified page
        """
        chunks = self.get_chunks_by_file_ids([document_id])
        page_chunks = [
            chunk for chunk in chunks
            if chunk.metadata.get('page_number') == page_number
        ]
        logger.info(f"📄 Retrieved {len(page_chunks)} chunks from page {page_number}")
        return page_chunks
    
    def get_chunks_by_page_range(
        self, 
        document_id: str, 
        start_page: int, 
        end_page: int
    ) -> List[Document]:
        """
        Get all chunks for a range of pages.
        
        Args:
            document_id: Document ID
            start_page: Starting page number
            end_page: Ending page number
        
        Returns:
            List of chunks from the specified page range
        """
        chunks = self.get_chunks_by_file_ids([document_id])
        page_chunks = [
            chunk for chunk in chunks
            if start_page <= chunk.metadata.get('page_number', 0) <= end_page
        ]
        logger.info(f"📄 Retrieved {len(page_chunks)} chunks from pages {start_page}-{end_page}")
        return page_chunks
    
    
    def get_document_pages(self, document_id: str) -> List[int]:
        """
        Get all page numbers for a document.
        
        Args:
            document_id: Document ID
        
        Returns:
            List of page numbers
        """
        chunks = self.get_chunks_by_file_ids([document_id])
        pages = sorted(set(
            chunk.metadata.get('page_number') 
            for chunk in chunks 
            if chunk.metadata.get('page_number') is not None
        ))
        logger.info(f"📄 Document {document_id} has {len(pages)} pages")
        return pages
    
    
    def get_page_count(self, document_id: str) -> int:
        """
        Get total number of pages for a document.
        
        Args:
            document_id: Document ID
        
        Returns:
            Total number of pages
        """
        chunks = self.get_chunks_by_file_ids([document_id])
        if chunks:
            return chunks[0].metadata.get('total_pages', 0)
        return 0
    
    
    def get_chunks_by_file_ids(self, file_ids: List[str]) -> List[Document]:
        """Retrieve chunks belonging to specific file IDs"""
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
    
    
    def get_all_chunks(self) -> List[Document]:
        """Get all indexed chunks"""
        return self.all_chunks
    
    
    def get_vector_store(self):
        """Get FAISS vector store"""
        return self.vector_store
    
    
    def get_bm25_retriever(self):
        """Get BM25 retriever"""
        return self.bm25_retriever
    
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics"""

        pages_by_doc = {}
        for chunk in self.all_chunks:
            doc_id = chunk.metadata.get('document_id')
            if doc_id:
                if doc_id not in pages_by_doc:
                    pages_by_doc[doc_id] = set()
                page_num = chunk.metadata.get('page_number')
                if page_num:
                    pages_by_doc[doc_id].add(page_num)
        
        total_pages = sum(len(pages) for pages in pages_by_doc.values())
        
        return {
            "faiss": {
                "total_vectors": self.vector_store.index.ntotal if self.vector_store else 0,
                "dimension": self.vector_store.index.d if self.vector_store else 0,
                "index_path": self.faiss_index_path,
                "index_type": self.faiss_index_type
            },
            "bm25": {
                "total_chunks": len(self.all_chunks),
                "index_path": self.bm25_index_path
            },
            "chunking": {
                "default_size": self.default_chunk_size,
                "default_overlap": self.default_chunk_overlap
            },
            "mmr": {
                "fetch_k": self.mmr_fetch_k,
                "lambda_mult": self.mmr_lambda_mult
            },
            "pages": {
                "total_pages": total_pages,
                "documents_with_pages": len(pages_by_doc),
                "pages_per_document": {k: len(v) for k, v in pages_by_doc.items()}
            }
        }
    
    
    def clear_indexes(self):
        """Clear all indexes"""
        self.vector_store = None
        self.bm25_retriever = None
        self.all_chunks = []
        logger.info("🗑️ Indexes cleared")