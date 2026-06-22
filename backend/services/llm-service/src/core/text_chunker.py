from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class TextChunker:
    def __init__(self):
        pass
    
    def chunk(self, doc: Document, chunk_size: int = 200, chunk_overlap: int = 20):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents([doc])
        print(f"===================={chunks}==========!!!!!!!!!!!!!!!!!!", flush=True)

        return chunks