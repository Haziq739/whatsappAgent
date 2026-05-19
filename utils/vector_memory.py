import chromadb
from sentence_transformers import SentenceTransformer
import datetime

# Dedicated embedding model for vector memory
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
client = chromadb.PersistentClient(path="./chroma_db")
# Separate collection to ensure we don't mix conversational history with strict store FAQs
collection = client.get_or_create_collection(name="user_long_term_memory")

def index_conversation_message(customer_number, role, message_text):
    """
    Saves a single message to long-term vector memory.
    """
    if not message_text or not str(message_text).strip():
        return
        
    try:
        # Create a unique document ID
        timestamp = datetime.datetime.now().isoformat()
        doc_id = f"{customer_number}_{timestamp}_{role}"
        
        # We store the role context with it to help retrieval
        contextual_text = f"[{role.upper()}] {message_text}"
        embedding = embedder.encode(contextual_text).tolist()
        
        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[contextual_text],
            metadatas=[{"customerNumber": customer_number, "role": role}]
        )
    except Exception as e:
        print(f"[ERROR] Error saving vector memory: {e}")

def query_long_term_memory(customer_number, query_text, limit=3):
    """
    Searches for messages similar to the current query for this specific customer.
    """
    if not query_text or not str(query_text).strip():
        return ""
        
    try:
        query_embedding = embedder.encode(query_text).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            where={"customerNumber": customer_number},
            n_results=limit
        )
        
        docs = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        
        # We want to filter out low-confidence matches. 
        # Since it's conversational, we'll keep the distance relatively generous 
        # compared to the strict FAQ guardrail.
        valid_history = []
        for doc, dist in zip(docs, distances):
            if dist < 28.0:
                valid_history.append(doc)
                
        return "\n".join(valid_history)
    except Exception as e:
        print(f"[ERROR] Error querying vector memory: {e}")
        return ""
