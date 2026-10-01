# LangChain Text Splitter: Document Chunking Best Practices

# Learn how to split documents for RAG using RecursiveCharacterTextSplitter with optimal chunk size strategies:
# 🔍 Text Splitting Strategies for RAG Applications:

#     • Optimal Chunk Size: Balance context preservation with retrieval precision for RAG
#     • Chunk Overlap: Maintain continuity for better retrieval augmented generation
#     • Semantic Boundaries: Split at natural breaks for document coherence
#     • RAG Optimization: Configure for your specific retrieval needs

# 📖 Code Explanation:

#     • RecursiveCharacterTextSplitter: Intelligently splits text at natural boundaries (paragraphs, sentences, words)
#     • chunk_size: Maximum characters per chunk - balance between context and LLM limits
#     • chunk_overlap: Overlapping text ensures context continuity between chunks
#     • Token-based splitting: Uses tiktoken to count actual LLM tokens instead of characters
#     • Custom separators: Define hierarchy for code, markdown, or domain-specific formats
#     • Metadata preservation: Each chunk inherits parent document metadata

# 🎯 Choosing Chunk Sizes:

# Small chunks (100-500 chars):

# ✅ More precise retrieval | ✅ Better for Q&A | ❌ May lose context | ❌ More chunks to manage

# Medium chunks (500-1500 chars):

# ✅ Good balance | ✅ Preserves paragraph context | ✅ Works with most LLMs | 🔧 Most common choice

# Large chunks (1500-4000 chars):

# ✅ Full context preserved | ✅ Good for summarization | ❌ Less precise retrieval | ❌ Higher token costs

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader

# 1. Basic Recursive Character Splitting
def demonstrate_basic_splitting():
    """Show basic text splitting with different configurations"""
    
    # Sample long text
    sample_text = """
    Artificial Intelligence (AI) is a broad field that encompasses machine learning, 
    natural language processing, computer vision, and robotics.
    
    Machine Learning is a subset of AI that focuses on algorithms that can learn 
    from and make predictions or decisions based on data. Deep learning, a subset 
    of machine learning, uses neural networks with multiple layers.
    
    Natural Language Processing (NLP) deals with the interaction between computers 
    and human language. It includes tasks like sentiment analysis, machine translation, 
    and text summarization.
    
    Computer Vision enables machines to interpret and understand visual information 
    from the world. Applications include image recognition, object detection, 
    and facial recognition.
    
    Robotics combines AI with mechanical engineering to create intelligent machines 
    that can perform tasks autonomously or with human guidance.
    """ * 5  # Repeat to make it longer
    
    # Basic splitter configuration
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,        # Maximum characters per chunk
        chunk_overlap=50,      # Characters to overlap between chunks
        length_function=len,   # Function to measure length
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_text(sample_text)
    
    print(f"Original text length: {len(sample_text)} characters")
    print(f"Number of chunks: {len(chunks)}")
    print()
    
    # Display chunks with overlap visualization
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        print(f"--- Chunk {i+1} ({len(chunk)} chars) ---")
        print(chunk[:150] + "..." if len(chunk) > 150 else chunk)
        print()
    
    return chunks

# 2. Token-Aware Splitting
def demonstrate_token_splitting():
    """Split text based on token count instead of characters"""
    import tiktoken
    
    # Use tiktoken for accurate token counting
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
    def tiktoken_len(text):
        tokens = encoding.encode(text)
        return len(tokens)
    
    # Token-based splitter
    token_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,           # Maximum tokens per chunk
        chunk_overlap=20,         # Token overlap
        length_function=tiktoken_len,  # Use token counting
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    sample_text = """
    Large Language Models (LLMs) are AI systems trained on vast amounts of text data. 
    They can generate human-like text, answer questions, translate languages, and 
    perform many other language-related tasks. Popular LLMs include GPT-4, Claude, 
    and PaLM. These models have billions of parameters and require significant 
    computational resources for training and inference.
    """ * 3
    
    chunks = token_splitter.split_text(sample_text)
    
    print("Token-based splitting:")
    print(f"Original text tokens: {tiktoken_len(sample_text)}")
    print(f"Number of chunks: {len(chunks)}")
    
    for i, chunk in enumerate(chunks):
        token_count = tiktoken_len(chunk)
        print(f"Chunk {i+1}: {token_count} tokens")
        print(f"Content: {chunk[:100]}...")
        print()
    
    return chunks

# 3. Document-Aware Splitting
def demonstrate_document_splitting():
    """Split documents while preserving metadata"""
    from langchain.schema import Document
    
    # Create sample documents with metadata
    documents = [
        Document(
            page_content="Python is a high-level programming language. " * 20,
            metadata={"source": "python_guide.txt", "chapter": "Introduction"}
        ),
        Document(
            page_content="Machine learning algorithms can be supervised or unsupervised. " * 25,
            metadata={"source": "ml_textbook.pdf", "chapter": "Algorithms", "page": 42}
        )
    ]
    
    # Document splitter preserves metadata
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=30
    )
    
    split_docs = text_splitter.split_documents(documents)
    
    print("Document splitting with metadata preservation:")
    print(f"Original documents: {len(documents)}")
    print(f"Split documents: {len(split_docs)}")
    print()
    
    for i, doc in enumerate(split_docs):
        print(f"--- Split Document {i+1} ---")
        print(f"Content: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}")
        print()
    
    return split_docs

# 4. Custom Separator Splitting
def demonstrate_custom_separators():
    """Use custom separators for specific document types"""
    
    # Code document with specific structure
    code_text = """
def process_data(data):
    '''Process input data'''
    cleaned_data = clean_data(data)
    return cleaned_data

class DataProcessor:
    def __init__(self, config):
        self.config = config
    
    def process(self, data):
        return self.transform(data)

# Configuration settings
CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.001
}
    """
    
    # Code-specific separators
    code_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        separators=[
            "\n\nclass ",      # Class definitions
            "\n\ndef ",       # Function definitions  
            "\n\n# ",         # Comments
            "\n\n",           # Double newlines
            "\n",              # Single newlines
            " ",                # Spaces
            ""                  # Characters
        ]
    )
    
    chunks = code_splitter.split_text(code_text)
    
    print("Code-aware splitting:")
    for i, chunk in enumerate(chunks):
        print(f"--- Code Chunk {i+1} ---")
        print(chunk)
        print()
    
    return chunks

# 5. Optimal Chunk Size Testing
def find_optimal_chunk_size(text, target_chunks=None):
    """Test different chunk sizes to find optimal configuration"""
    chunk_sizes = [100, 200, 500, 1000, 2000]
    results = []
    
    for size in chunk_sizes:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=size // 10  # 10% overlap
        )
        
        chunks = splitter.split_text(text)
        avg_length = sum(len(chunk) for chunk in chunks) / len(chunks)
        
        results.append({
            'chunk_size': size,
            'num_chunks': len(chunks),
            'avg_length': avg_length,
            'shortest': min(len(chunk) for chunk in chunks),
            'longest': max(len(chunk) for chunk in chunks)
        })
    
    print("Chunk size optimization results:")
    print(f"{'Size':<6} {'Chunks':<7} {'Avg Len':<8} {'Min':<6} {'Max':<6}")
    print("-" * 40)
    
    for result in results:
        print(f"{result['chunk_size']:<6} "
              f"{result['num_chunks']:<7} "
              f"{result['avg_length']:<8.0f} "
              f"{result['shortest']:<6} "
              f"{result['longest']:<6}")
    
    return results

# Example usage
if __name__ == "__main__":
    print("=== Text Splitting Demonstrations ===\n")
    
    # Run demonstrations
    basic_chunks = demonstrate_basic_splitting()
    print("\n" + "="*50 + "\n")
    
    # Uncomment if tiktoken is installed
    # token_chunks = demonstrate_token_splitting()
    # print("\n" + "="*50 + "\n")
    
    doc_chunks = demonstrate_document_splitting()
    print("\n" + "="*50 + "\n")
    
    code_chunks = demonstrate_custom_separators()
    print("\n" + "="*50 + "\n")
    
    # Test with sample text
    sample_text = "Lorem ipsum dolor sit amet. " * 100
    find_optimal_chunk_size(sample_text)
