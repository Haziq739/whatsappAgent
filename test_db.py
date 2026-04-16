import chromadb
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="business_Knowledge")

query = "S25 Ultra price?"
print(f"Query: {query}")
query_embedding = embedder.encode(query).tolist()
results = collection.query(query_embeddings=[query_embedding], n_results=3)
res1 = results.get("documents", [[]])[0]

query = "Redmi 14C kitne ka hai?"
print(f"Query: {query}")
query_embedding = embedder.encode(query).tolist()
results = collection.query(query_embeddings=[query_embedding], n_results=3)
res2 = results.get("documents", [[]])[0]

with open("test_db_output.txt", "w", encoding="utf-8") as f:
    f.write("S25 Ultra price? Results:\n")
    for r in res1:
        f.write(r + "\n")
    f.write("\nRedmi 14C kitne ka hai? Results:\n")
    for r in res2:
        f.write(r + "\n")


