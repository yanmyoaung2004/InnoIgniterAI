import chromadb
from sentence_transformers import SentenceTransformer

# Initialize Hugging Face embedding model
hf_model = SentenceTransformer("all-MiniLM-L6-v2")

class HuggingFaceEmbeddingFunction:
    def __call__(self, input):
        return hf_model.encode(input).tolist()
    def name(self):
        return "huggingface-all-MiniLM-L6-v2"
    
embedding_function = HuggingFaceEmbeddingFunction()

# Initialize ChromaDB client and collection
chroma_client = chromadb.Client(settings=chromadb.Settings(
    persist_directory="app/chroma_db"))
collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    embedding_function=embedding_function
)

def add_documents(docs, ids=None):
    """
    Add documents to the vector DB.
    docs: list of strings
    ids: list of unique string IDs (optional)
    """
    if ids is None:
        ids = [str(i) for i in range(len(docs))]
    collection.add(documents=docs, ids=ids)

def search_db(query: str, top_k: int = 3) -> list:
    """
    Search the vector DB for relevant context using Hugging Face embeddings.
    Returns a list of context strings.
    """
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    return results.get("documents", [])[0] if results.get("documents") else []