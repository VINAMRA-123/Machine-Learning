# Loading Different Document Types

# Master how to split documents for RAG using LangChain document loader. Each loader implements document chunking best practices for retrieval augmented generation applications:
# 🔍 LangChain Document Processing Tutorial Components:

#     • TextLoader: Process text files for RAG with optimal chunk size
#     • PyPDFLoader: PDF to vector conversion with page-aware splitting
#     • WebBaseLoader: Web content processing for retrieval augmented generation
#     • DirectoryLoader: Batch document processing with text splitting strategies

# 📖 Code Explanation:

#     • TextLoader: Loads plain text files with automatic encoding detection to handle various character sets
#     • PyPDFLoader: Extracts text from PDFs page by page, preserving page boundaries in metadata
#     • WebBaseLoader: Fetches and cleans HTML content from web pages, removing tags and scripts
#     • DirectoryLoader: Batch processes multiple files with glob patterns for efficient loading
#     • Error Handling: Comprehensive try-catch blocks to handle missing files or format issues

# 💡 Expected Output:

# Loaded 1 text documents
# First doc preview: This is a sample document for testing LangChain loaders.
# This is a sample document for testing LangChain loaders...
# Metadata: {'source': 'sample_document.txt'}

# ⚠️ Common Issues and Solutions:

#     • Encoding errors: Use encoding='utf-8' or autodetect_encoding=True
#     • Large PDFs: Use PyPDFLoader.load_and_split() for automatic chunking
#     • Web timeouts: Add requests_kwargs={"timeout": 30} to WebBaseLoader
    • Memory issues: Process files in batches rather than loading all at once

from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    WebBaseLoader,
    DirectoryLoader
)
import os

# 1. Loading Text Files
def load_text_documents():
    """Load plain text files with proper encoding"""
    text_loader = TextLoader("sample_document.txt", encoding="utf-8")
    text_docs = text_loader.load()
    
    print(f"Loaded {len(text_docs)} text documents")
    print(f"First doc preview: {text_docs[0].page_content[:200]}...")
    print(f"Metadata: {text_docs[0].metadata}")
    
    return text_docs

# 2. Loading PDF Files
def load_pdf_documents():
    """Load PDF files page by page"""
    pdf_loader = PyPDFLoader("research_paper.pdf")
    pdf_docs = pdf_loader.load()
    
    print(f"Loaded {len(pdf_docs)} PDF pages")
    for i, doc in enumerate(pdf_docs[:3]):  # Show first 3 pages
        print(f"Page {i+1} preview: {doc.page_content[:150]}...")
        print(f"Page {i+1} metadata: {doc.metadata}")
    
    return pdf_docs

# 3. Loading Web Pages
def load_web_documents():
    """Load and clean web content"""
    urls = [
        "https://python.langchain.com/docs/get_started/introduction",
        "https://python.langchain.com/docs/modules/data_connection"
    ]
    
    web_loader = WebBaseLoader(urls)
    web_docs = web_loader.load()
    
    print(f"Loaded {len(web_docs)} web pages")
    for doc in web_docs:
        print(f"URL: {doc.metadata.get('source', 'Unknown')}")
        print(f"Content preview: {doc.page_content[:200]}...")
    
    return web_docs

# 4. Batch Loading with DirectoryLoader
def load_directory_documents():
    """Load all documents from a directory"""
    # Load all .txt and .pdf files from a directory
    loader = DirectoryLoader(
        "documents/",
        glob="**/*.{txt,pdf}",
        loader_cls=TextLoader,  # Default loader for text files
        loader_kwargs={'autodetect_encoding': True}
    )
    
    docs = loader.load()
    print(f"Loaded {len(docs)} documents from directory")
    
    # Group by file type
    file_types = {}
    for doc in docs:
        ext = doc.metadata['source'].split('.')[-1]
        file_types[ext] = file_types.get(ext, 0) + 1
    
    print(f"File types: {file_types}")
    return docs

# 5. Custom Loader with Error Handling
def load_documents_safely(file_paths):
    """Load documents with comprehensive error handling"""
    loaded_docs = []
    failed_files = []
    
    for file_path in file_paths:
        try:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                print(f"Unsupported file type: {file_path}")
                continue
                
            docs = loader.load()
            loaded_docs.extend(docs)
            print(f"✅ Successfully loaded: {file_path}")
            
        except Exception as e:
            print(f"❌ Failed to load {file_path}: {str(e)}")
            failed_files.append((file_path, str(e)))
    
    print(f"\nSummary:")
    print(f"Successfully loaded: {len(loaded_docs)} documents")
    print(f"Failed files: {len(failed_files)}")
    
    if failed_files:
        print("\nFailed files:")
        for file_path, error in failed_files:
            print(f"  {file_path}: {error}")
    
    return loaded_docs, failed_files

# Example usage
if __name__ == "__main__":
    # Create sample files for testing
    with open("sample_document.txt", "w", encoding="utf-8") as f:
        f.write("This is a sample document for testing LangChain loaders.\n" * 10)
    
    # Test different loaders
    text_docs = load_text_documents()
    
    # For PDF and web loading, you'll need actual files/URLs
    # pdf_docs = load_pdf_documents()
    # web_docs = load_web_documents()
    
    # Clean up
    if os.path.exists("sample_document.txt"):
        os.remove("sample_document.txt")
