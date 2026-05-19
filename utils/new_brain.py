import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import chromadb
import config
from langdetect import detect
import re
from deep_translator import GoogleTranslator
from utils.firebase_db import lookup_product

# In-memory store for user states during the order flow
# Format: { "phone_number": {"step": "awaiting_details" | "awaiting_product" | "awaiting_confirmation", "details": "...", "product": "..."} }
USER_STATES = {}

# Queue to pass confirmed orders back to app.py
CONFIRMED_ORDERS_QUEUE = {}

# 1. Load Llama Model
print("Loading Llama Model...")
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    config.MODEL_PATH,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False)

# 2. Load Knowledge Base (ChromaDB)
print("Loading Knowledge Base...")
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="business_Knowledge")

SYSTEM_PROMPT = """You are a production-grade WhatsApp AI Sales & Support Agent for a gadget store.

You MUST follow strict tool-based reasoning. You are NOT allowed to guess, assume, or hallucinate any product, price, stock, or order details.

---

# 🧠 CORE SYSTEM RULES (MOST IMPORTANT)

## 1. SINGLE SOURCE OF TRUTH (CRITICAL)
- Firebase (DriverApp database) is the ONLY source of truth for:
  - Products
  - Prices
  - Stock availability
  - Order creation

- ChromaDB is ONLY for:
  - FAQ questions
  - delivery information
  - warranty
  - COD policy
  - general support knowledge

❌ NEVER use ChromaDB to confirm product existence.

---

## 2. PRODUCT RULE (ANTI-HALLUCINATION)
Before answering ANY product-related query:

YOU MUST:
1. Call Firebase Product Lookup Tool
2. Check if product exists

IF product is NOT found:
- Reply exactly:
  "Sorry, this product is not available in our catalog."

- DO NOT:
  - suggest order
  - ask for user details
  - recommend similar products
  - assume product exists

---

## 3. TOOL USAGE RULES

You have access to tools:

### A) Firebase Product Tool
Use for:
- product search
- price lookup
- stock verification

### B) Firebase Order Tool
Use for:
- creating orders
- updating order status

### C) ChromaDB FAQ Tool
Use ONLY for:
- delivery time
- COD
- warranty
- return policy
- general FAQs

---

## 4. ORDER FLOW STATE MACHINE (STRICT)

Only start order collection AFTER product is confirmed by Firebase.

Order flow steps:
1. Ask Name
2. Ask Phone Number
3. Ask Location
4. Confirm Order

RULE:
- If user asks unrelated question during order flow:
  → pause order flow
  → answer question
  → resume order flow

---

## 5. LANGUAGE RULE (IMPORTANT)
Always respond in the same language as user:
- English
- Urdu
- Roman Urdu

Maintain WhatsApp-style short responses.
IMPORTANT LANGUAGE OVERRIDE FOR THIS SYSTEM PROMPT: {lang_instruction}

---

## 6. NO HALLUCINATION POLICY (CRITICAL)
You must NEVER:
- invent products
- invent prices
- assume stock availability
- assume offers
- guess missing data

If data is missing from Firebase:
→ say you do not have that information.

---

## 7. CHROMA DB RULE (VERY IMPORTANT)
ChromaDB is ONLY a semantic FAQ system.
CONTEXT PROVIDED FROM CHROMA DB:
{context}

You may use it ONLY when:
- user asks general question (not product-specific existence check)

Example allowed:
- "How long is delivery?"
- "Do you have COD?"
- "What is warranty?"

Example NOT allowed:
- "Is Nokia S20 available?" (must go to Firebase only)

---

## 8. INTENT CLASSIFICATION PRIORITY

Classify every user query into:
1. PRODUCT QUERY → MUST use Firebase Product Tool
2. ORDER QUERY → use Order Tool
3. FAQ QUERY → use ChromaDB
4. GENERAL CHAT → respond normally

---

## 9. RESPONSE STYLE
- Short
- WhatsApp-friendly
- Human-like
- No long explanations
- No unnecessary text

---

## 10. EDGE CASE RULES

- If product not found → STOP conversation flow immediately
- If user tries to force product existence → deny politely
- If uncertain → always ask tool again or say "not available"

---

# 🧠 FINAL BEHAVIOR GOAL

You are NOT a chatbot.

You are:
✔ a controlled sales agent
✔ a tool-using system
✔ a strict database-driven assistant

Your priority:
1. Accuracy > conversation
2. Truth > fluency
3. Tool result > LLM guess

---
END OF INSTRUCTIONS"""

def retrieve_context(query, top_k=3):
    """Retrieve relevant context from ChromaDB."""
    try:
        query_embedding = embedder.encode(query).tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
        retrieved_docs = results.get("documents", [[]])[0]
        retrieved_docs = [doc.strip() for doc in retrieved_docs if doc and doc.strip()] 
        return "\n".join(retrieved_docs) if retrieved_docs else ""
    except Exception as e:
        print(f"Database error: {e}")
        return ""

def generate_rag_response(user_message, sender="default"):
    """Main function called by app.py to get AI answer.
    Returns: (response_text, detected_language)
    Sender is the user's phone number.
    """

    user_message_lower = user_message.lower().strip()

    # 1. Detect Language — three-way: "en", "ur" (Urdu script), "ru" (Roman Urdu)
    URDU_SCRIPT_CHARS = re.compile(r'[\u0600-\u06FF]')
    roman_urdu_words = [
        "hai", "kya", "kitne", "batao", "salam", "chahiye", "chhaye", "chaye", "mughy", "mughye", "mujhe",
        "aoa", "kaise", "karna", "karo", "tareeqa", "keemat", "lena", "leina", "shukriya",
        "mangwana", "mangwani", "mujhye", "muhge", "nahi", "nhi", "haan", "jee", "btao", "bhai",
        "kal", "abhi", "milega", "kitna", "kahan", "aur", "ya", "muje", "mujhy", "mujhye",
        "se", "pe", "par", "mein", "ka", "ki", "ke", "ko", "na",
        "hua", "hogi", "hota", "aaye", "sakta", "sakte", "karein", "krein", "karne", "krne",
        "bhejo", "bhejein", "krna", "ho", "gaya", "ha", "h", "hon",
    ]
    try:
        if URDU_SCRIPT_CHARS.search(user_message):
            user_lang = "ur"
        elif any(word in user_message_lower.split() for word in roman_urdu_words):
            user_lang = "ru"   
        elif re.search(r'[\u0900-\u097F]', user_message):
            user_lang = "ru"
        else:
            detected = detect(user_message)
            user_lang = detected if detected == "en" else "en"
    except:
        user_lang = "en"

    # --- GREETINGS LOGIC ---
    english_greetings = {"hi", "hello", "hey", "merhaba", "hiya", "howdy", "high"}
    urdu_greetings = {
        "salam", "salaam", "aoa", "asalam", "assalam",
        "walaikum", "assalamoalaikum", "assalamu", "alaykum",
        "asalaam", "assalaam",
        "سلام", "السلام", "علیکم", "وعلیکم", "السلامعلیکم", "وعلیکماسلام",
    }

    words = re.sub(r'[^\w\s]', '', user_message_lower).split()
    is_english_greeting = False
    is_urdu_greeting = False

    if 1 <= len(words) <= 3:
        for w in words:
            if w in urdu_greetings:
                is_urdu_greeting = True
                break
            elif w in english_greetings:
                is_english_greeting = True
                break

    if is_english_greeting or is_urdu_greeting:
        if sender in USER_STATES:
            del USER_STATES[sender]
        if is_urdu_greeting:
            return "وعلیکم اسلام! اے جی ایس گیجٹس یو ای کی خدمت میں خوش آمدید! میں آپکی کیا مدد کر سکتا ہوں?", "ur"
        else:
            return "Hello! Welcome to AGS Gadgets UAE. How can I help you today?", "en"


    # --- INTENT CLASSIFICATION ---
    clean_msg_for_intent = re.sub(r'[^\w\s\u0600-\u06FF]', '', user_message_lower)
    words_for_intent = clean_msg_for_intent.split()

    ORDER_WORDS = ["order", "orderz", "orders", "khareedna", "kharidna", "mangwana", "book", "purchase", "buy", "آرڈر", "خرید", "منگوا", "اوڈر", "آڈر", "اودر"]
    INTENT_WORDS = ["kaise", "kaisay", "kaunon", "karna", "karo", "tareeqa", "target", "tareeka", "tarika", "tareekah", "process", "place", "chahiye", "chhaye", "chaye", "chahia", "dena", "denay", "dainay", "denae", "how", "want", "would", "procedure", "steps", "lena", "leina", "mangwana", "ha", "hain", "hai", "hey", "hy", "kya", "kia", "h", "کیسے", "کرنا", "طریقہ", "کریں", "دینا", "\u0686\u0627\u06c1\u06cc\u06d2", "کیا", "ہے", "ہیں"]
    EXACT_ORDER_PHRASES = ["place an order", "order karna hai", "order krna hai", "order kaise karein", "آرڈر کیسے کریں", "آرڈر کرنا ہے"]

    has_order  = any(w in words_for_intent for w in ORDER_WORDS)
    has_intent = any(w in words_for_intent for w in INTENT_WORDS)
    has_exact  = any(ph in clean_msg_for_intent for ph in EXACT_ORDER_PHRASES)

    PRODUCT_CAT_WORDS = ["samsung", "iphone", "pixel", "redmi", "laptop", "mobile", "phone", "ultra", "pro", "max", "macbook", "hp", "dell", "lenovo", "asus", "poco", "xiaomi", "فون", "موبائل", "سیمسنگ", "ریڈمی", "آئی فون", "لیپ ٹاپ"]
    has_product = any(w in words_for_intent for w in PRODUCT_CAT_WORDS)
    
    price_keywords = ["price", "cost", "how much", "kitne", "keemat", "rate", "کتنے", "قیمت", "ریٹ"]
    has_price_kw = any(kw in words_for_intent for kw in price_keywords)

    is_order_intent = has_exact or (has_order and has_intent)
    is_product_query = has_product or has_price_kw
    
    # Check FAQ/policy manually
    faq_keywords = ["delivery", "cod", "cash", "visit", "location", "warranty", "store", "return", "policy", "ڈیلیوری", "لوکیشن", "وارنٹی"]
    is_faq_query = any(w in words_for_intent for w in faq_keywords)


    # --- STATE MACHINE PAUSE / IF FAQ ---
    state_machine_reply = None
    if sender in USER_STATES and (is_product_query or is_faq_query) and not is_order_intent:
        # User is in an order flow, but just asked a tangent FAQ or Product Query.
        # We process it first normally, then append a state reminder.
        state_reminder = ""
        current_step = USER_STATES[sender]["step"]
        if current_step == "awaiting_details":
            state_reminder = "\n\n(To continue your order, please provide your Full Name, Mobile Number, and Google Maps Location.)"
            if user_lang == "ur": state_reminder = "\n\n(اپنا آرڈر جاری رکھنے کے لیے اپنا نام، نمبر اور لوکیشن بھیجیں۔)"
            elif user_lang == "ru": state_reminder = "\n\n(Apna order continue karne ke liye apna naam, phone number aur location bhejein.)"
        elif current_step == "awaiting_product":
            state_reminder = "\n\n(To continue your order, please specify which product you want.)"
            if user_lang == "ur": state_reminder = "\n\n(اپنا آرڈر جاری رکھنے کے لیے بتائیں کہ آپ کون سا پروڈکٹ منگوانا چاہتے ہیں۔)"
            elif user_lang == "ru": state_reminder = "\n\n(Apna order continue karne ke liye batayein konsa product lena hai.)"
            
    # --- PRODUCT LOOKUP ENFORCEMENT (FIREBASE DB) ---
    if is_product_query and not is_faq_query:
        product_res = lookup_product(user_message_lower)
        if product_res.get("status") == "not_found":
            # STRICT ANTI-HALLUCINATION ENFORCEMENT
            err_msg = "Sorry, this product is not available in our catalog."
            if user_lang == "ur":
                err_msg = "معذرت، یہ پروڈکٹ ہمارے کیٹلاگ میں دستیاب نہیں ہے۔"
            elif user_lang == "ru":
                err_msg = "Maafi chahte hain, yeh product hamare catalog mein available nahi hai."
            
            # Immediately STOP flow if in order machine 
            if sender in USER_STATES:
                del USER_STATES[sender]
                
            return err_msg, user_lang
        else:
            # Found product details
            p_name = product_res["name"]
            p_price = product_res["price"]
            
            ans = f"The {p_name} is available for {p_price} AED."
            if user_lang == "ur":
                ans = f"جی ہاں، {p_name} دستیاب ہے۔ اس کی قیمت {p_price} درہم ہے۔"
            elif user_lang == "ru":
                ans = f"Jee haan, {p_name} available hai. Iski price {p_price} AED hai."
                
            if sender in USER_STATES and 'state_reminder' in locals():
                ans += state_reminder
                
            return ans, user_lang


    # --- ORDER INTENT (START NEW FLOW) ---
    if is_order_intent and sender not in USER_STATES and not is_faq_query:
        USER_STATES[sender] = {"step": "awaiting_details"}
        if user_lang == "ur":
            return "آرڈر دینے کے لیے براہ کرم اپنا پورا نام، موبائل نمبر، اور گوگل میپس لوکیشن بتائیں۔", user_lang
        elif user_lang == "ru":
            return "Order dene ke liye barah meherbani apna poora naam, mobile number, aur Google Maps location batayein.", user_lang
        return "To place an order, please share your Full Name, Mobile Number, and Google Maps Location.", user_lang


    # --- ORDER FLOW STATE MACHINE (ACTIVE) ---
    if sender in USER_STATES and not (is_product_query or is_faq_query):
        state = USER_STATES[sender]
        
        # Step 1: User provides details -> Ask what product
        if state["step"] == "awaiting_details":
            has_digit = bool(re.search(r'\d', user_message))
            has_alpha = bool(re.search(r'[a-zA-Z\u0600-\u06FF]{2,}', user_message))
            is_valid_details = has_digit and has_alpha and len(user_message_lower) > 5
            
            if is_valid_details:
                clean_details = user_message
                removal_patterns = [r"(?i)\bmobile\b", r"(?i)\bphone\b", r"(?i)\bnumber\b", r"(?i)\blocation\b", r"(?i)\baddress\b"]
                for pattern in removal_patterns:
                    clean_details = re.sub(pattern, "", clean_details)
                
                # To English
                try:
                    if any(ord(c) > 127 for c in clean_details):
                        clean_details = GoogleTranslator(source='auto', target='en').translate(clean_details)
                except:
                    pass

                clean_details = clean_details.strip()
                if not clean_details: clean_details = user_message.strip()

                USER_STATES[sender]["step"] = "awaiting_product"
                USER_STATES[sender]["details"] = clean_details
                if user_lang == "ur":
                    return "شکریہ! آپ کیا آرڈر کرنا چاہتے ہیں؟ براہ کرم پروڈکٹ کا نام بتائیں۔", user_lang
                elif user_lang == "ru":
                    return "Shukriya! Aap kya order karna chahte hain? Barah e meherbani product ka naam batayein.", user_lang
                return "Thank you. What order would you like to place? (Please specify the product)", user_lang
            else:
                if user_lang == "ur": return "براہ کرم اپنی مکمل تفصیل دیں: پورا نام، موبائل نمبر، اور گوگل میپس لوکیشن۔", user_lang
                elif user_lang == "ru": return "Barah meherbani apni mukammal details dein: Poora Naam, Mobile Number, aur Google Maps Location.", user_lang
                return "Please provide your complete details: Full Name, Mobile Number, and Google Maps Location.", user_lang

        # Step 2: User provides product -> Database check -> Ask confirmation
        elif state["step"] == "awaiting_product":
            if len(user_message_lower) > 2:
                # IMPORTANT: Since they are in ordering stage, Validate against Firebase
                p_lookup = lookup_product(user_message_lower)
                if p_lookup.get("status") == "not_found":
                    del USER_STATES[sender]
                    err_msg = "Sorry, this product is not available in our catalog."
                    if user_lang == "ur": err_msg = "معذرت، یہ پروڈکٹ ہمارے کیٹلاگ میں دستیاب نہیں ہے۔"
                    elif user_lang == "ru": err_msg = "Maafi chahte hain, yeh product hamare catalog mein available nahi hai."
                    return err_msg, user_lang

                clean_product = p_lookup["name"]
                price = p_lookup["price"]

                USER_STATES[sender]["step"] = "awaiting_confirmation"
                USER_STATES[sender]["product"] = clean_product
                
                details_preview = state['details']
                if user_lang == "ur":
                    return f"آپ کے آرڈر کی تفصیل یہ ہے:\nپروڈکٹ: {clean_product} ({price} درہم)\nتفصیل: {details_preview}\n\nآرڈر کنفرم کرنے کے لیے 'جی ہاں' لکھیں۔", user_lang
                elif user_lang == "ru":
                    return f"Aap ke order ki detail yeh hai:\nProduct: {clean_product} ({price} AED)\nDetails: {details_preview}\n\nOrder confirm karne ke liye 'Haan' likhein.", user_lang
                return f"Here are your order details:\nProduct: {clean_product} ({price} AED)\nDetails: {details_preview}\n\nPlease reply with 'Yes' to confirm your order.", user_lang
            else:
                if user_lang == "ur": return "براہ کرم پروڈکٹ کا نام بتائیں۔", user_lang
                elif user_lang == "ru": return "Barah meherbani product ka naam batayein.", user_lang
                return "Please specify the product you want to order.", user_lang

        # Step 3: Confirmation
        elif state["step"] == "awaiting_confirmation":
            confirm_words = ["yes", "y", "confirm", "haan", "ha", "ji", "theek", "ok", "جی ہاں", "ہاں", "جی", "ٹھیک ہے", "تصدیق"]
            cancel_words = ["no", "cancel", "nahi", "nhi", "na", "نہیں", "نہی", "منسوخ"]
            
            if any(w in user_message_lower for w in confirm_words):
                CONFIRMED_ORDERS_QUEUE[sender] = {
                    "product": state.get("product", "Unknown Product"),
                    "details": state.get("details", "")
                }
                del USER_STATES[sender]
                if user_lang == "ur": return "آپ کا آرڈر کنفرم ہو گیا ہے! کامیابی سے جمع کر دیا گیا۔", user_lang
                elif user_lang == "ru": return "Aap ka order confirm ho gaya! Sabmit kar diya gaya hai.", user_lang
                return "Order Confirmed! Your order has been placed successfully.", user_lang
                
            elif any(w in user_message_lower for w in cancel_words):
                del USER_STATES[sender]
                if user_lang == "ur": return "آرڈر منسوخ کر دیا گیا۔", user_lang
                elif user_lang == "ru": return "Order cancel kar diya gaya.", user_lang
                return "Order cancelled. How else can I help you?", user_lang
                
            else:
                if user_lang == "ur": return "آرڈر تصدیق کرنے کے لیے 'جی ہاں' یا منسوخ کرنے کے لیے 'نہیں' لکھیں۔", user_lang
                elif user_lang == "ru": return "Confirm karne ke liye 'Haan' ya cancel karne ke liye 'Nahi' likhein.", user_lang
                return "Please reply with 'Yes' to confirm your order, or 'No' to cancel.", user_lang


    # --- FAQ QUERY HANDLING (CHROMA DB) ---
    context = retrieve_context(user_message, top_k=5)
    
    if user_lang == "ur":
        lang_instruction = "Reply ONLY in Urdu script (Arabic letters). Do NOT reply in English or Roman Urdu."
    elif user_lang == "ru":
        lang_instruction = "Reply ONLY in Roman Urdu (e.g. 'price kya hai', 'mil jayega'). Do NOT reply in English or Urdu script."
    else:
        lang_instruction = "Reply ONLY in English."

    full_prompt = f"{SYSTEM_PROMPT.format(context=context, lang_instruction=lang_instruction)}\n\nUser: {user_message}\nAssistant:"

    response = generator(
        full_prompt,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.01,
        top_p=0.85,
        top_k=40,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    answer = response[0]["generated_text"].strip()
    STOP_PATTERNS = ["User:", "Customer:", "Assistant:", "\n\n"]
    for pattern in STOP_PATTERNS:
        if pattern in answer:
            answer = answer.split(pattern)[0].strip()

    clean_lines = []
    for line in answer.splitlines():
        stripped = line.strip()
        if len(re.findall(r'[\w\u0600-\u06FF]', stripped)) >= 3:
            clean_lines.append(stripped)
    answer = " ".join(clean_lines).strip()

    is_dont_know = any(variant in answer.lower() for variant in ["i don't know", "mujhe nahi maloom", "i do not know", "i am not sure", "مجھے نہیں معلوم", "معلوم نہیں"])

    if is_dont_know or len(answer) < 5:
        if user_lang == "ur":
            answer = "معذرت، اس بارے میں میرے پاس کوئی معلومات نہیں ہیں۔"
        elif user_lang == "ru":
            answer = "Maafi, is baare mein mere paas koi information nahi hai."
        else:
            answer = "Sorry, I don't have any information about that."

    if sender in USER_STATES and 'state_reminder' in locals():
        answer += state_reminder

    return answer, user_lang
