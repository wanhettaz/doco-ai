# Project Notes - AI Assistant Development

## Overview
Building a local RAG system for document retrieval using llama-index and Ollama. The goal is to create a private AI assistant that can search through personal notes and documentation without sending data to external APIs.

## Technical Stack
- **LLM**: Llama 3.2 (running locally via Ollama)
- **Embeddings**: BAAI/bge-small-en-v1.5 from HuggingFace
- **Vector Store**: ChromaDB for persistent storage
- **Framework**: llama-index for orchestration
- **Interface**: Gradio for web UI

## Architecture Decisions
The system uses semantic search rather than keyword matching. Documents are chunked into 512-token segments with 50-token overlap to maintain context across boundaries.

Vector embeddings are 384-dimensional, which provides a good balance between accuracy and performance for local systems.

## Performance Considerations
- Retrieval typically takes 100-300ms depending on document count
- LLM generation adds 2-5 seconds with local models
- ChromaDB uses HNSW indexing for O(log n) search complexity

## Next Steps
- Implement metadata filtering for file types
- Add reranking for better retrieval quality
- Experiment with query decomposition for complex questions
- Consider adding Hyde (hypothetical document embeddings) for improved recall