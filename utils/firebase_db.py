import firebase_admin
from firebase_admin import credentials, firestore
import datetime
import os
import re
from deep_translator import GoogleTranslator
from google.cloud.firestore_v1.base_query import FieldFilter
from utils.vector_memory import index_conversation_message

# --- Local chat log directory ---
CHAT_LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chat_logs")
os.makedirs(CHAT_LOGS_DIR, exist_ok=True)

# Initialize Firebase
db = None
try:
    # Look for serviceAccountKey.json in the project root directory
    cred_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "serviceAccountKey.json")
    if os.path.exists(cred_path):
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("[SUCCESS] Firebase initialized successfully!")
    else:
        print("[WARN] serviceAccountKey.json not found! Firebase ops disabled until added.")
except Exception as e:
    print(f"[ERROR] Error initializing Firebase: {e}")

def get_or_create_conversation(customer_number, agent_number):
    if not db:
        return None
        
    try:
        conversations_ref = db.collection('conversation_messages')
        # Check for active conversation
        query = conversations_ref.where(filter=FieldFilter('customerNumber', '==', customer_number)).where(filter=FieldFilter('status', '==', 'active')).limit(1).get()
        
        if query:
            return query[0].id
        else:
            # Create a new conversation
            timestamp = int(datetime.datetime.now().timestamp())
            conversation_id = f"{customer_number}_{timestamp}"
            
            conversations_ref.document(conversation_id).set({
                'customerNumber': customer_number,
                'agentNumber': agent_number,
                'createdAt': firestore.SERVER_TIMESTAMP,
                'lastMessage': '',
                'status': 'active',
                'messages_log': []
            })
            return conversation_id
    except Exception as e:
        print(f"[ERROR] Error getting/creating conversation: {e}")
        return None

def _log_to_file(conversation_id, sender_type, message_text):
    """Append a message to the local chat log file in real-time."""
    try:
        log_path = os.path.join(CHAT_LOGS_DIR, f"{conversation_id}.txt")
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        role = "User" if sender_type == "customer" else "Assistant"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp_str}] {role}: {message_text}\n")
    except Exception as e:
        print(f"[ERROR] Error writing local chat log: {e}")


def store_message(customer_number, agent_number, message_text, sender_type):
    """
    sender_type should be 'customer' or 'agent'.
    Stores the message in:
      1. Firebase 'messages' collection (individual message doc)
      2. Firebase 'conversations' document (appended to messages_log array)
      3. Local text file (chat_logs/{conversation_id}.txt)
    """
    if not db:
        return
        
    try:
        conversation_id = get_or_create_conversation(customer_number, agent_number)
        if not conversation_id:
            return
            
        if sender_type == 'customer':
            sender_num = customer_number
            receiver_num = agent_number
        else:
            sender_num = agent_number
            receiver_num = customer_number

        timestamp_now = datetime.datetime.now().isoformat()
        
        # 1. Append message to conversation_messages document's messages_log array
        conv_doc_ref = db.collection('conversation_messages').document(conversation_id)
        conv_doc_ref.update({
            'lastMessage': message_text,
            'lastMessageAt': firestore.SERVER_TIMESTAMP,
            'messages_log': firestore.ArrayUnion([{
                'sender': sender_type,
                'text': message_text,
                'timestamp': timestamp_now
            }])
        })

        # 3. Write to local text file
        _log_to_file(conversation_id, sender_type, message_text)

        # 4. Save to Long-Term Vector Memory
        index_conversation_message(customer_number, sender_type, message_text)

    except Exception as e:
        print(f"[ERROR] Error storing message: {e}")

def get_recent_history(customer_number, current_query=None, limit=5):
    """
    Fetches the last N messages from the active session of this customer.
    Returns a list of dicts suitable for Llama 3: [{'role': 'user'/'assistant', 'content': '...'}]
    """
    if not db: return []
    try:
        conversations_ref = db.collection('conversation_messages')
        query = conversations_ref.where(filter=FieldFilter('customerNumber', '==', customer_number)).where(filter=FieldFilter('status', '==', 'active')).limit(1).get()
        if not query:
            return []
            
        doc_data = query[0].to_dict()
        messages_log = doc_data.get('messages_log', [])
        
        messages_list = []
        for msg in messages_log:
            role = "user" if msg.get('sender') == 'customer' else "assistant"
            text = msg.get('text', '')
            
            # Skip empty messages
            if not text.strip(): continue
            # If the last message in DB is EXACTLY the query we are processing right now, skip it from history
            if role == "user" and current_query and text.strip() == current_query.strip():
                continue
                
            messages_list.append({"role": role, "content": text})
            
        # Keep only the last N messages
        return messages_list[-limit:]
    except Exception as e:
        print(f"[ERROR] Error fetching history: {e}")
        return []

def lookup_product(query):
    """
    Checks if a product exists in Firebase based on query.
    Returns -> dict(status="found", name, price) OR dict(status="not_found")
    """
    if not db:
        return {"status": "not_found", "reason": "db_not_initialized"}
        
    try:
        products_ref = db.collection('products')
        docs = products_ref.get()
        query_cln = query.lower().strip()
        
        best_match = None
        highest_score = 0
        
        # Simple flexible scoring based on keyword matching
        words = re.sub(r'[^\w\s]', '', query_cln).split()
        
        # Identify number sequences in query (e.g. ['25', '16'])
        query_digits = set(re.findall(r'\d+', query_cln))
        
        for doc in docs:
            p_data = doc.to_dict()
            p_name = p_data.get('name', '').lower()
            p_name_words = p_name.split()
            
            p_digits = set(re.findall(r'\d+', p_name))
            
            # STRICT FILTER: If the user specified a number (like 16 for A16),
            # the product MUST contain that number if the product has numbers.
            if query_digits and p_digits:
                if not query_digits.intersection(p_digits):
                    continue # Skip products with completely different model numbers
            
            score = 0
            for w in words:
                if len(w) > 2 and w in p_name:
                    score += 1
            
            # Exact number/model match gives big boost
            for w in words:
                if any(c.isdigit() for c in w):
                    if w in p_name_words:
                        score += 5
                    elif any(w in pw or pw in w for pw in p_name_words):
                        score += 3
                        
            if score > highest_score and score >= 2: # Requires at least some relevance
                highest_score = score
                best_match = p_data
                
        if best_match:
            return {"status": "found", "name": best_match.get("name"), "price": best_match.get("price"), "stock_status": best_match.get("stock_status")}
            
        return {"status": "not_found"}
    except Exception as e:
        print(f"[ERROR] Error looking up product: {e}")
        return {"status": "not_found"}

def create_order(customer_number, raw_details, product_name, customer_name="", customer_phone="", customer_location=""):
    if not db:
        return
        
    try:
        agent_number = "Agent" # Can be updated if we pass agent number down
        conversation_id = get_or_create_conversation(customer_number, agent_number)

        # Helper to translate anything non-English to English
        def to_english(text):
            try:
                # If it has Urdu/Arabic characters or Hindi (Devanagari) characters, translate it
                if re.search(r'[\u0600-\u06FF\u0900-\u097F]', text):
                    return GoogleTranslator(source='auto', target='en').translate(text)
                return text
            except:
                return text

        # Use pre-parsed fields if provided, otherwise fall back to parsing raw_details
        if customer_name and customer_name not in ("Customer", "Not provided"):
            name_final = customer_name
        else:
            # Basic extraction from raw_details (typically comma-separated: Name, Number, Location)
            parts = [p.strip() for p in raw_details.split(',')]
            name_final = parts[0] if len(parts) > 0 else raw_details
        
        if customer_location and customer_location != "Not provided":
            location_final = customer_location
        else:
            parts = [p.strip() for p in raw_details.split(',')]
            location_final = parts[-1] if len(parts) > 1 else "Unknown (Check Details)"
        
        if customer_phone and customer_phone != "Not provided":
            phone_final = customer_phone
        else:
            # Try to extract phone from raw_details
            phone_match = re.search(r'(\+?\d[\d\s\-]{7,15}\d)', raw_details)
            phone_final = re.sub(r'[\s\-]', '', phone_match.group(1)) if phone_match else customer_number

        # Translate all fields to English for consistent storage
        print(f"🔄 Translating order details to English...")
        name_en = to_english(name_final)
        product_name_en = to_english(product_name)
        location_en = to_english(location_final)

        # CLEAN Name from extra symbols/digits
        name_en = re.sub(r'[^a-zA-Z\s]', '', name_en).strip().title()
        
        # Fallback if empty after translation/cleaning
        if not name_en:
            name_en = "Customer"

        orders_ref = db.collection('orders_chatbot')
        order_data = {
            'conversationId': conversation_id,
            'customerName': name_en,
            'customerNumber': phone_final,
            'productName': product_name_en.title(),
            'location': location_en.title(),
            'status': 'confirmed',
            'timestamp': firestore.SERVER_TIMESTAMP
        }
        
        doc_ref = orders_ref.add(order_data)
        print(f"[SUCCESS] Order saved successfully to Firebase with ID: {doc_ref[1].id}")
        print(f"[ORDER DATA] Name: {name_en}, Phone: {phone_final}, Product: {product_name_en}, Location: {location_en}")
        
    except Exception as e:
        print(f"[ERROR] Error creating order in Firebase: {e}")
