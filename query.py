from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

def setup_llm_and_embeddings():
    """same setup as ingest - need consistent models"""
    Settings.llm = Ollama(
        model="llama3.2",
        request_timeout=120.0
    )
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

def load_index():
    """
    load the existing index from disk
    no re-indexing needed, just connect to chromadb
    """
    setup_llm_and_embeddings()
    
    db = chromadb.PersistentClient(path="./data/chroma_db")
    chroma_collection = db.get_or_create_collection("quickstart")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # load from existing vector store
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=StorageContext.from_defaults(vector_store=vector_store)
    )
    return index

def query(question: str, similarity_top_k: int = 3):
    """
    the RAG pipeline:
    1. embed the question
    2. find top_k most similar chunks from vector db
    3. stuff them into context for the llm
    4. llm generates answer based on retrieved context
    """
    index = load_index()
    
    # query engine handles the retrieval + generation
    query_engine = index.as_query_engine(
        similarity_top_k=similarity_top_k,  # how many chunks to retrieve
        response_mode="compact"  # uses less tokens, good for local models
    )
    
    response = query_engine.query(question)
    return response

if __name__ == "__main__":
    # test it
    result = query("what are the main topics in my documents?")
    print(result)