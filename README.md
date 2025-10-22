# doco-ai

A local RAG (Retrieval-Augmented Generation) system for semantic search and question-answering across personal documents. Built with llama-index, Ollama, and ChromaDB—no external APIs, fully private.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview

doco-ai implements a private AI assistant that can search, summarise, and answer questions about your local documents using semantic search and local LLM inference. Perfect for querying personal notes, documentation, meeting transcripts, or any text-based knowledge base without sending data to external services.

**Key Features:**
- 🔒 Fully local—no data leaves your machine
- 🚀 Fast semantic search using vector embeddings
- 📚 Support for markdown, text, and PDF files
- 💬 Clean web interface for natural language queries
- 🧠 Persistent vector storage for instant retrieval

## Architecture

```
User Query
    ↓
Gradio Interface (app.py)
    ↓
Query Engine (query.py)
    ↓
Vector Search (ChromaDB) → Top-K Relevant Chunks
    ↓
LLM Generation (Ollama) → Contextual Answer
```

**Technical Stack:**
- **LLM**: Llama 3.2 (or Qwen 2.5) via Ollama
- **Embeddings**: BAAI/bge-small-en-v1.5 (384-dimensional)
- **Vector Store**: ChromaDB with HNSW indexing
- **Framework**: llama-index for orchestration
- **Interface**: Gradio web UI
- **Chunking**: 256 tokens with 50-token overlap

## Prerequisites

- Python 3.10 or higher
- Ollama installed ([installation guide](https://ollama.com/download))
- At least 8GB RAM (16GB recommended for larger models)

## Installation

### 1. Install Ollama

```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# Download from https://ollama.com/download
```

### 2. Pull an LLM Model

```bash
ollama pull llama3.2
# or
ollama pull qwen2.5:7b
```

### 3. Clone and Setup

```bash
git clone https://github.com/yourusername/doco-ai.git
cd doco-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir docs data
```

## Usage

### 1. Add Your Documents

Place your files in the `docs/` directory:
```bash
docs/
├── meeting_notes.md
├── project_docs.txt
└── research_paper.pdf
```

Supported formats: `.md`, `.txt`, `.pdf`

### 2. Index Your Documents

```bash
python ingest.py
```

This will:
- Read all documents from `docs/`
- Split them into semantic chunks
- Generate embeddings
- Store vectors in ChromaDB

**Expected output:**
```
setting up models...
initialising vector store...
loading documents from ./docs...
found 5 documents
creating index (this takes a min)...
Parsing nodes: 100%|████████| 24/24
Generating embeddings: 100%|████████| 24/24
✓ indexing complete. vectors stored in ./data/chroma_db
```

**Note**: The `data/` folder is gitignored as it contains your indexed documents. After cloning this repo, you'll need to run `python ingest.py` to generate your own vector database.

### 3. Launch the Interface

```bash
python app.py
```

Navigate to `http://localhost:7860` in your browser.

### 4. Query Your Documents

Example queries:
- "Summarise the main points from the Q4 meeting"
- "What embedding model am I using and why?"
- "Explain the difference between security groups and NACLs"
- "What are the next action items mentioned in my notes?"

## Project Structure

```
doco-ai/
├── ingest.py           # Document indexing script
├── query.py            # Retrieval and query logic
├── app.py              # Gradio web interface
├── requirements.txt    # Python dependencies
├── .gitignore          # Git ignore rules
├── docs/               # Your documents go here
├── data/               # Vector database storage (auto-generated, gitignored)
│   └── chroma_db/
└── README.md
```

## Security & Privacy

doco-ai is designed with privacy as a priority:

- ✅ All processing happens locally—no external API calls
- ✅ Documents and vectors stay on your machine
- ✅ Ollama runs on localhost (no network exposure)
- ✅ ChromaDB stores data on disk with standard filesystem permissions

**Security considerations:**

1. **Network access**: By default, the Gradio interface binds to `0.0.0.0`, making it accessible on your local network. To restrict to localhost only:
```python
# In app.py
demo.launch(
    server_name="127.0.0.1",  # localhost only
    server_port=7860
)
```

2. **Authentication**: Add basic auth if exposing the interface:
```python
demo.launch(auth=("username", "password"))
```

3. **Gradio sharing**: Never set `share=True` with sensitive documents—it creates a public tunnel through Gradio's servers.

4. **First-time setup**: The embedding model downloads from HuggingFace on first run (cached locally afterward).

## Configuration

### Adjust Chunk Size

In `ingest.py`, modify the `SentenceSplitter`:

```python
text_splitter = SentenceSplitter(
    chunk_size=256,      # Smaller = more precise, less context
    chunk_overlap=50     # Overlap prevents info loss at boundaries
)
```

### Change LLM Model

In both `ingest.py` and `query.py`:

```python
Settings.llm = Ollama(
    model="qwen2.5:7b",  # or llama3.2, mistral, phi-3, etc.
    request_timeout=120.0
)
```

### Modify Retrieval Settings

In `query.py`:

```python
query_engine = index.as_query_engine(
    similarity_top_k=3,      # Number of chunks to retrieve
    response_mode="compact"  # or "refine", "tree_summarize"
)
```

## Performance

Typical performance on M1/M2 Mac or modern x86 CPU:

| Operation | Time |
|-----------|------|
| Document indexing | 1-3 seconds per document |
| Vector search | 50-200ms |
| LLM generation | 2-5 seconds (local model) |
| Total query time | ~3-7 seconds |

## Extending the System

### Add Metadata Filtering

```python
# In ingest.py, add metadata to documents
from datetime import datetime

for doc in documents:
    doc.metadata["indexed_date"] = datetime.now().isoformat()
    doc.metadata["file_type"] = doc.metadata["file_name"].split(".")[-1]

# In query.py, filter by metadata
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter

filters = MetadataFilters(filters=[
    MetadataFilter(key="file_type", value="md")
])
query_engine = index.as_query_engine(filters=filters)
```

### Add Reranking

```bash
pip install llama-index-postprocessor-cohere
```

```python
from llama_index.postprocessor.cohere import CohereRerank

reranker = CohereRerank(top_n=3)
query_engine = index.as_query_engine(
    node_postprocessors=[reranker]
)
```

### Enable Conversation History

Modify `app.py` to use `history` parameter:

```python
def chat(message, history):
    # Build context from history
    context = "\n".join([f"User: {h[0]}\nAssistant: {h[1]}" for h in history])
    full_query = f"{context}\nUser: {message}"
    
    response = query(full_query)
    return str(response)
```

## Git Ignore Rules

The following are automatically excluded from version control:

```
# Virtual environment
venv/
env/

# Vector database (generated locally)
data/
*.sqlite3

# Python cache
__pycache__/
*.pyc

# OS files
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/

# Optional: exclude personal documents
# docs/
```

**Note**: The `docs/` folder can optionally be gitignored if you're indexing sensitive personal documents. Consider keeping example/sample documents in the repo for testing purposes.

## Troubleshooting

### "Settings is not defined"
Ensure you've imported Settings:
```python
from llama_index.core import Settings
```

### "Cannot import ChromaVectorStore"
Install the integration:
```bash
pip install llama-index-vector-stores-chroma
```

### Ollama connection errors
Make sure Ollama is running:
```bash
ollama serve
```

### Low retrieval quality
- Increase `similarity_top_k` for more context
- Adjust chunk size (larger for more context, smaller for precision)
- Try a different embedding model
- Add a reranker for better relevance

### Tokenizers parallelism warning
Add to the top of your scripts:
```python
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```

## Advanced Use Cases

- **Code documentation assistant**: Index your codebase and API docs
- **Meeting notes search**: Query across months of meeting transcripts
- **Research paper assistant**: Analyse and compare academic papers
- **Personal knowledge base**: Second brain for all your notes and ideas
- **Lab documentation**: Query experimental logs and procedures

## Roadmap

- [ ] Add support for more file types (DOCX, HTML)
- [ ] Implement query decomposition for complex questions
- [ ] Add citation display showing source documents
- [ ] Build CLI interface for terminal usage
- [ ] Support for multiple languages
- [ ] Implement Hyde (hypothetical document embeddings)
- [ ] Add vector store migration tools

## Contributing

Contributions welcome! Please open an issue or submit a PR.

## License

MIT License - feel free to use this project however you want.

## Acknowledgements

- [llama-index](https://github.com/run-llama/llama_index) for the RAG framework
- [Ollama](https://ollama.com) for local LLM inference
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Gradio](https://gradio.app) for the web interface

---

**doco-ai**: Your documents, your AI, your machine.