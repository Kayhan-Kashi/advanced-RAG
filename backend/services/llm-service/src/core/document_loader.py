from typing import List, Optional, Tuple, Dict, Any
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from injector import inject
import logging
import os
import fitz 

logger = logging.getLogger(__name__)


class DocumentLoader:
    @inject
    def __init__(self):
        self._documents: List[Document] = []
        self._file_path: Optional[str] = None
        self._images: List[str] = []  # Store image paths
        self._image_paths: List[str] = []  # Alias for images
        self._pages: List[Dict[str, Any]] = []  # Store page metadata
        self._page_texts: List[str] = []  # Store individual page texts
    
    def load(self, file_path: str, encoding: str = "utf-8") -> 'DocumentLoader':
        """Load a text document"""
        try:
            loader = TextLoader(file_path, encoding=encoding)
            self._documents = loader.load()
            self._file_path = file_path
            logger.info(f"✅ Loaded text document from {file_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load document: {e}")
            self._documents = []
        return self
    
    def load_pdf(self, file_path: str, output_dir: str = "pages", dpi: int = 300) -> 'DocumentLoader':
        """
        Load a PDF file, extract images and text with PAGE NUMBERS preserved
        """
        doc = None
        try:
            if not os.path.exists(file_path):
                logger.error(f"❌ PDF file not found: {file_path}")
                self._documents = []
                self._pages = []
                self._page_texts = []
                return self
            
            logger.info(f"📖 Loading PDF from: {file_path} (size: {os.path.getsize(file_path)} bytes)")
            
            os.makedirs(output_dir, exist_ok=True)
            
            doc = fitz.open(file_path)
            logger.info(f"📖 PDF opened: {len(doc)} pages")
            
            self._images = []
            self._image_paths = []
            self._pages = []   # Reset page texts
            
            page_documents = []
            extracted_text = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                page_text = page.get_text()
                extracted_text.append(page_text)
                self._page_texts.append(page_text)
                
                page_info = {
                    "page_number": page_num + 1,
                    "text": page_text,
                    "char_count": len(page_text),
                    "word_count": len(page_text.split()),
                    "image_path": None
                }
                self._pages.append(page_info)
                
                pix = page.get_pixmap(dpi=dpi)
                img_path = f"{output_dir}/page_{page_num + 1}.png"
                pix.save(img_path)
                self._images.append(img_path)
                self._image_paths.append(img_path)
                page_info["image_path"] = img_path
                
                page_doc = Document(
                    page_content=page_text,
                    metadata={
                        "source": file_path,
                        "type": "pdf",
                        "page_number": page_num + 1, 
                        "total_pages": len(doc),
                        "image_path": img_path,
                        "char_count": len(page_text),
                        "word_count": len(page_text.split()),
                    }
                )
                page_documents.append(page_doc)
                
                logger.debug(f"📄 Extracted page {page_num + 1}: {len(page_text)} chars")
            
            full_text = "\n\n".join(extracted_text)
            
            self._documents = page_documents
            

            combined_metadata = {
                "source": file_path,
                "type": "pdf",
                "total_pages": len(doc),
                "page_count": len(doc),
                "page_numbers": [p["page_number"] for p in self._pages],
                "page_range": f"1-{len(doc)}",
                "image_paths": self._images,
                "has_page_metadata": True,
            }
            
            self._file_path = file_path
            
            logger.info(f"✅ Loaded PDF from {file_path}")
            logger.info(f"   📄 {len(doc)} pages with individual page metadata")
            logger.info(f"   🖼️  {len(self._images)} images extracted")
            logger.info(f"   📝 Total text: {len(full_text)} characters")
            
            if self._pages:
                sample = self._pages[0]
                logger.info(f"   📋 Sample: Page {sample['page_number']} - {sample['char_count']} chars")
            
        except fitz.fitz.FileDataError as e:
            logger.error(f"❌ PDF file is corrupted or invalid: {e}")
            self._documents = []
            self._images = []
            self._pages = []
            self._page_texts = []
        except Exception as e:
            logger.error(f"❌ Failed to load PDF: {e}")
            logger.exception("Detailed traceback:")
            self._documents = []
            self._images = []
            self._pages = []
            self._page_texts = []
        finally:
            if doc is not None:
                doc.close()
        
        return self
    
    def load_pdf_with_text_only(self, file_path: str) -> 'DocumentLoader':
        """
        Load only text from PDF (no image extraction) with PAGE NUMBERS preserved
        """
        try:
            doc = fitz.open(file_path)
            
            self._pages = []
            self._page_texts = []
            page_documents = []
            extracted_text = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                extracted_text.append(page_text)
                self._page_texts.append(page_text)
                
                page_info = {
                    "page_number": page_num + 1,
                    "text": page_text,
                    "char_count": len(page_text),
                    "word_count": len(page_text.split()),
                }
                self._pages.append(page_info)
                
                page_doc = Document(
                    page_content=page_text,
                    metadata={
                        "source": file_path,
                        "type": "pdf",
                        "page_number": page_num + 1,  
                        "total_pages": len(doc),
                        "char_count": len(page_text),
                        "word_count": len(page_text.split()),
                    }
                )
                page_documents.append(page_doc)
            
            self._documents = page_documents
            
            self._file_path = file_path
            doc.close()
            
            logger.info(f"✅ Loaded PDF text from {file_path}")
            logger.info(f"   📄 {len(doc)} pages with page metadata")
            
        except Exception as e:
            logger.error(f"❌ Failed to load PDF text: {e}")
            self._documents = []
            self._pages = []
            self._page_texts = []
        return self
    
    
    def get_pages(self) -> List[Dict[str, Any]]:
        """Get all pages with their metadata"""
        return self._pages
    
    
    def get_page_count(self) -> int:
        """Get total number of pages"""
        return len(self._pages)
    
    
    def get_page(self, page_number: int) -> Optional[Dict[str, Any]]:
        """Get specific page by number"""
        for page in self._pages:
            if page["page_number"] == page_number:
                return page
        return None
    
    
    def get_page_text(self, page_number: int) -> Optional[str]:
        """Get text of specific page"""
        for page in self._pages:
            if page["page_number"] == page_number:
                return page["text"]
        return None
    
    
    def get_page_documents(self) -> List[Document]:
        """Get documents split by page with metadata"""
        return self._documents
    
    
    def get_page_document(self, page_number: int) -> Optional[Document]:
        """Get document for specific page"""
        for doc in self._documents:
            if doc.metadata.get("page_number") == page_number:
                return doc
        return None
    
    
    def get_page_range(self, start_page: int, end_page: int) -> List[Document]:
        """Get documents for a range of pages"""
        return [
            doc for doc in self._documents
            if start_page <= doc.metadata.get("page_number", 0) <= end_page
        ]
    
    
    @property
    def documents(self) -> List[Document]:
        """Get loaded documents (individual pages with metadata)"""
        return self._documents
    
    
    @property
    def text(self) -> str:
        """Get combined text content of all pages"""
        return "\n\n".join([doc.page_content for doc in self._documents])
    
    
    @property
    def images(self) -> List[str]:
        """Get list of extracted image paths (from PDF)"""
        return self._images
    
    
    @property
    def image_paths(self) -> List[str]:
        """Alias for images"""
        return self._images
    
    
    @property
    def is_loaded(self) -> bool:
        """Check if document is loaded"""
        return len(self._documents) > 0
    
    
    @property
    def metadata(self) -> dict:
        """Get metadata of the first page (for backward compatibility)"""
        if self._documents:
            return self._documents[0].metadata
        return {}
    
    
    @property
    def page_numbers(self) -> List[int]:
        """Get list of all page numbers"""
        return [page["page_number"] for page in self._pages]
    
    
    @property
    def total_pages(self) -> int:
        """Get total number of pages"""
        return len(self._pages)