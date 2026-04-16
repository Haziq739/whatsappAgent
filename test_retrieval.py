from sentence_transformers import SentenceTransformer
import chromadb

embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="business_Knowledge")

def test_retrieval(query):
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=5)
    docs = results.get("documents", [[]])[0]
    print(f"\nQUERY: {query}")
    for i, doc in enumerate(docs):
        print(f"[{i}] {doc}")

test_retrieval("What products do you have?")
test_retrieval("Ap ka pass kon kon se products hain?")
