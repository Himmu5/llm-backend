from db.db_config import get_db
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from urllib.parse import quote
import os

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent
UPLOADS_DIR = PROJECT_ROOT / "uploads"
DOCS_DIR = PROJECT_ROOT / "docs"
DEFAULT_PDF = DOCS_DIR / "resume.pdf"

# GCS bucket name (same as in myapp.py)
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "himanshu-rag")

def get_splitted_data(fileName=None):
    if fileName is None:
        file_path = str(DEFAULT_PDF)
    elif fileName.startswith(('http://', 'https://')):
        # Full URL - use as-is
        file_path = fileName
    elif fileName.startswith('/'):
        # Absolute path - use as-is
        file_path = fileName
    else:
        # Check local folders first
        uploads_path = UPLOADS_DIR / fileName
        docs_path = DOCS_DIR / fileName
        
        if uploads_path.exists():
            file_path = str(uploads_path)
        elif docs_path.exists():
            file_path = str(docs_path)
        else:
            # File not found locally - try GCS URL (URL-encode the filename)
            encoded_filename = quote(fileName, safe='')
            file_path = f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{encoded_filename}"
    
    print(f"Loading PDF from: {file_path}")
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    if not docs:
        raise ValueError(f"No content found in PDF: {file_path}")
    
    # Check if documents have any text
    total_text = "".join([doc.page_content for doc in docs])
    if not total_text.strip():
        raise ValueError(f"PDF has no extractable text (may be image-based): {file_path}")
    
    print(f"Loaded {len(docs)} pages with {len(total_text)} characters")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    if not splits:
        raise ValueError(f"No text chunks created from PDF: {file_path}")
    
    print(f"Created {len(splits)} text chunks")
    return splits
 

def get_retriever(fileName):
    """Get retriever for a single file"""
    try:
        data = get_splitted_data(fileName)
        print(f"📄 Sample content from PDF: {data[0].page_content[:500]}...")
        vector_db = get_db(data)
        retriever = vector_db.as_retriever()
        return retriever
    except Exception as e:
        print(f"Error creating retriever: {e}")
        raise


def get_retriever_multi(fileNames: list):
    """Get retriever for multiple files"""
    try:
        all_splits = []
        for fileName in fileNames:
            print(f"📄 Processing: {fileName}")
            try:
                splits = get_splitted_data(fileName)
                all_splits.extend(splits)
                print(f"   ✅ Added {len(splits)} chunks from {fileName}")
            except Exception as e:
                print(f"   ⚠️ Skipping {fileName}: {e}")
                continue
        
        if not all_splits:
            raise ValueError("No content found in any of the provided files")
        
        print(f"📚 Total chunks from {len(fileNames)} files: {len(all_splits)}")
        print(f"📄 Sample content: {all_splits[0].page_content[:300]}...")
        
        vector_db = get_db(all_splits)
        retriever = vector_db.as_retriever()
        return retriever
    except Exception as e:
        print(f"Error creating multi-file retriever: {e}")
        raise
