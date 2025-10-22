from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import StorageContext



def setup_llm_and_embeddings():
    """
    configure the brains of the operation
    - ollama connects to your local llm (make sure it's running: ollama serve)
    - embeddings turn text into numbers that capture meaning
    """

    Settings.llm =  Ollama(
        model = "llama3.2", # or whatever you pulled
        request_timeout = 120.0
    )

    # this embedding model is solid and free
    # it converts text chunks into 384-dimensional vectors
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

def ingest_documents():
    """
    The actual index magic happens here
    """
    print("setting up models....")
    setup_llm_and_embeddings()

    #chromadb is the vector database of choice - store embeddings for fast retrieval
    print("initialising vector store.....")
    db = chromadb.PersistentClient(path="./data/chroma_db")
    chroma_collection = db.get_or_create_collection("quickstart")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    #Read everything in ./docs recursively
    print("loading documents from ./docs.....")
    documents = SimpleDirectoryReader(
        "./docs",
        recursive=True,
        required_exts=[".md", ".txt", ".pdf"] #filter for these file types
    ).load_data()

    print(f"found {len(documents)} documents")

    text_splitter = SentenceSplitter(
        chunk_size=256,
        chunk_overlap=50
    )

    # the actual indexing - chunks docs, embeds them, stores in chromadb
    # chunk_size=256 means each piece is ~256 tokens
    # chunk_overlap=50 gives context between chunks so nothing gets lost
    print("creating index (this takes a min)...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[text_splitter],
        show_progress=True
    )

    print("✓ indexing complete. vectors stored in ./data/chroma_db")
    return index

if __name__ == "__main__":
    ingest_documents()