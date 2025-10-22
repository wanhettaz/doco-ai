# RAG System Concepts

## What is RAG?
Retrieval-Augmented Generation combines the power of large language models with external knowledge retrieval. Instead of relying solely on the model's training data, RAG systems fetch relevant information at query time.

## How Vector Search Works

### Embeddings
Text is converted into high-dimensional vectors (typically 384 to 1536 dimensions) that capture semantic meaning. Similar concepts end up close together in vector space, even if they use different words.

Example: "machine learning" and "artificial intelligence" would have high cosine similarity despite sharing no words.

### Similarity Metrics
- **Cosine Similarity**: Measures angle between vectors (most common)
- **Euclidean Distance**: Straight-line distance in vector space
- **Dot Product**: Combines magnitude and direction

## Chunking Strategies
Documents must be split into manageable pieces:
- **Fixed-size chunking**: Simple but can break context
- **Sentence-based**: Respects natural boundaries
- **Semantic chunking**: Uses embeddings to find natural break points

Overlap between chunks (typically 10-20%) prevents information loss at boundaries.

## The RAG Pipeline
1. Query embedding: Convert user question to vector
2. Similarity search: Find top-k most relevant chunks
3. Context assembly: Combine retrieved chunks
4. Generation: LLM produces answer using context
5. (Optional) Reranking: Refine retrieved results

## Advanced Techniques

### Multi-Query
Generate multiple variations of the user's question to improve recall.

### Hyde (Hypothetical Document Embeddings)
Generate a hypothetical answer to the question, embed it, then search. Counterintuitive but effective.

### Query Decomposition
Break complex questions into simpler sub-queries, answer each, then synthesize.