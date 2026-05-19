import requests
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import chromadb
import config
import re
import json
import time

print("Loading Cloud-based FAQ Tool (Hugging Face API)...")

# Local vector engine for your private business data remains active
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="business_Knowledge")

SYSTEM_PROMPT = """You are a production-grade WhatsApp AI Sales & Support Agent for AGS Gadgets UAE.

You MUST follow strict tool-based reasoning. You are NOT allowed to guess, assume, or hallucinate any product, price, stock, or order details.

---

# 🧠 CORE SYSTEM RULES (MOST IMPORTANT)

## 1. SINGLE SOURCE OF TRUTH (CRITICAL)
- Firebase (DriverApp database) is the ONLY source of truth for:
  - Products, Prices, Stock availability, Order creation

- ChromaDB is ONLY for:
  - FAQ questions, delivery information, warranty, COD policy, general support knowledge

❌ NEVER use ChromaDB to confirm product existence.

---

## 2. STRICT DOMAIN GUARDRAIL (MOST CRITICAL RULE)
- You are ONLY permitted to discuss AGS Gadgets, electronic products, and store policies.
- If a user asks an unrelated question, you MUST respond EXACTLY:
  "I am only programmed to assist with AGS Gadgets and store-related queries."

---

## 3. RESPONSE QUALITY RULE (VERY IMPORTANT)
- You MUST NEVER reply with just one word like "Yes", "No", "Ok".
- Every reply MUST be a complete, helpful sentence.
- Your answer MUST directly address the user's question using the CONTEXT provided below.

---

## 4. CHROMA DB CONTEXT (USE THIS TO ANSWER)
CONTEXT PROVIDED FROM CHROMA DB:
{context}

---

## 5. LANGUAGE RULE (IMPORTANT)
Always respond in the same language as user:
- English / Urdu / Roman Urdu
IMPORTANT LANGUAGE OVERRIDE: {lang_instruction}

---

## 6. NO HALLUCINATION POLICY (CRITICAL)
If data is missing from context: say you do not have that information.
"""

def retrieve_context(query, top_k=5, distance_threshold=25.0):
    try:
        query_embedding = embedder.encode(query).tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
        retrieved_docs = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        valid_docs = [doc.strip() for doc, dist in zip(retrieved_docs, distances) if dist < distance_threshold]
        return "\n".join(valid_docs) if valid_docs else ""
    except Exception as e:
        print(f"[TOOL ERROR] Context retrieval failed: {e}")
        return ""

def faq_lookup_tool(user_message, user_lang="en", tool_results=None, history=None, long_term_context=""):
    """
    Checks ChromaDB and then calls Hugging Face Inference API for the final answer.
    """
    print(f"[TOOL CALLED] FAQTool (HF Cloud) -> Resolving intent for: '{user_message}'")

    context = retrieve_context(user_message, top_k=5)
    
    if user_lang == "ur":
        lang_instruction = "Reply ONLY in Urdu script (Arabic letters). Do NOT reply in English or Roman Urdu."
    elif user_lang == "ru":
        lang_instruction = "Reply ONLY in Roman Urdu. Do NOT reply in English or Urdu script."
    else:
        lang_instruction = "Reply ONLY in English."

    tool_info = ""
    if tool_results: tool_info += f"\n\nPREVIOUS TOOL RESULTS:\n{tool_results}"
    if long_term_context: tool_info += f"\n\nPAST USER MEMORY (RELEVANT):\n{long_term_context}"

    system_content = f"{SYSTEM_PROMPT.format(context=context, lang_instruction=lang_instruction)}{tool_info}"
    
    # Structure messages for Llama 3 Instruct template
    # Hugging Face Inference API often expects a specific formatting or a simple prompt string
    messages = [{"role": "system", "content": system_content}]
    if history: messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    # Standard Quality Guard
    ONE_WORD_JUNK = {"yes", "no", "ok", "okay", "sure"}
    
    # Call Hugging Face API using the official client (handles endpoints correctly)
    try:
        client_hf = InferenceClient(model=config.LLM_MODEL, token=config.HUGGINGFACEHUB_API_TOKEN)
        
        # Use the chat_completion method which is the "v1" equivalent in the library
        response = client_hf.chat_completion(
            messages=messages,
            max_tokens=300,
            temperature=0.1
        )
        
        answer = response.choices[0].message.content.strip()
        print(f"[LLM RAW (Cloud)] {answer}")

    except Exception as e:
        print(f"[API ERROR] Hugging Face failed for model {config.LLM_MODEL}: {e}")
        
        # Fallback to another reputable model if the first choice fails
        FALLBACK_MODEL = "HuggingFaceH4/zephyr-7b-beta"
        if config.LLM_MODEL != FALLBACK_MODEL:
            print(f"[INFO] Attempting fallback to {FALLBACK_MODEL}...")
            try:
                fallback_client = InferenceClient(model=FALLBACK_MODEL, token=config.HUGGINGFACEHUB_API_TOKEN)
                response = fallback_client.chat_completion(
                    messages=messages,
                    max_tokens=300,
                    temperature=0.1
                )
                answer = response.choices[0].message.content.strip()
                print("[SUCCESS] Fallback successful.")
                return answer
            except Exception as fe:
                print(f"[API ERROR] Fallback also failed: {fe}")
        
        answer = "Sorry, I am having trouble connecting to my brain. Please try again."

    print(f"[LLM RAW (Cloud)] {answer}")

    if answer.lower().strip().rstrip('.!?') in ONE_WORD_JUNK:
        if context.strip():
            context_lines = [line.strip() for line in context.strip().splitlines() if line.strip()]
            informative_lines = [line for line in context_lines if not line.endswith('?')]
            expansion = informative_lines[0][:200] if informative_lines else context_lines[0][:200]
            answer = f"{answer.title()}, {expansion.lower()}" if "yes" in answer.lower() else f"{answer.title()}. {expansion}"

    return answer
