import firebase_admin
from firebase_admin import credentials, firestore
import datetime
import os
import re
from deep_translator import GoogleTranslator

# Initialize Firebase
db = None
try:
    # Look for serviceAccountKey.json in the project root directory
    cred_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "serviceAccountKey.json")
    if os.path.exists(cred_path):
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("✅ Firebase initialized successfully!")
    else:
        print("⚠️ WARNING: serviceAccountKey.json not found! Firebase ops disabled until added.")
except Exception as e:
    print(f"❌ Error initializing Firebase: {e}")

def get_or_create_conversation(customer_number, agent_number):
    if not db:
        return None
        
    try:
        conversations_ref = db.collection('conversations')
        # Check for active conversation
        query = conversations_ref.where('customerNumber', '==', customer_number).where('status', '==', 'active').limit(1).get()
        
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
                'status': 'active'
            })
            return conversation_id
    except Exception as e:
        print(f"❌ Error getting/creating conversation: {e}")
        return None

def store_message(customer_number, agent_number, message_text, sender_type):
    """
    sender_type should be 'customer' or 'agent'
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

        messages_ref = db.collection('messages')
        messages_ref.add({
            'conversationId': conversation_id,
            'sender': sender_type,
            'senderNumber': sender_num,
            'receiverNumber': receiver_num,
            'messageText': message_text,
            'timestamp': firestore.SERVER_TIMESTAMP
        })
        
        db.collection('conversations').document(conversation_id).update({
            'lastMessage': message_text
        })
    except Exception as e:
        print(f"❌ Error storing message: {e}")

def create_order(customer_number, raw_details, product_name):
    if not db:
        return
        
    try:
        agent_number = "Agent" # Can be updated if we pass agent number down
        conversation_id = get_or_create_conversation(customer_number, agent_number)
        
        # Basic extraction from raw_details (typically comma-separated: Name, Number, Location)
        parts = [p.strip() for p in raw_details.split(',')]
        customer_name = parts[0] if len(parts) > 0 else raw_details
        location = parts[-1] if len(parts) > 1 else "Unknown (Check Details)"

        # Helper to translate anything non-English to English
        def to_english(text):
            try:
                # If it has Urdu/Arabic characters or Hindi (Devanagari) characters, translate it
                if re.search(r'[\u0600-\u06FF\u0900-\u097F]', text):
                    return GoogleTranslator(source='auto', target='en').translate(text)
                return text
            except:
                return text

        # Translate all fields to English for consistent storage
        print(f"🔄 Translating order details to English...")
        customer_name_en = to_english(customer_name)
        product_name_en = to_english(product_name)
        location_en = to_english(location)

        # CLEAN Name from extra symbols/digits
        customer_name_en = re.sub(r'[^a-zA-Z\s]', '', customer_name_en).strip().title()
        
        # Fallback if empty after translation/cleaning
        if not customer_name_en:
            customer_name_en = "Customer"

        orders_ref = db.collection('orders_chatbot')
        order_data = {
            'conversationId': conversation_id,
            'customerName': customer_name_en,
            'customerNumber': customer_number,
            'productName': product_name_en.title(),
            'location': location_en.title(),
            'status': 'confirmed',
            'timestamp': firestore.SERVER_TIMESTAMP
        }
        
        doc_ref = orders_ref.add(order_data)
        print(f"✅ Order saved successfully to Firebase with ID: {doc_ref[1].id}")
        
    except Exception as e:
        print(f"❌ Error creating order in Firebase: {e}")
