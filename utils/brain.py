import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import chromadb
import config
from langdetect import detect
import re

# In-memory store for user states during the order flow
# Format: { "phone_number": {"step": "awaiting_details" | "awaiting_product" | "awaiting_confirmation", "details": "...", "product": "..."} }
USER_STATES = {}

# Queue to pass confirmed orders back to app.py
CONFIRMED_ORDERS_QUEUE = {}

# 1. Load Llama Model
print("🧠 Loading Llama Model...")
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    config.MODEL_PATH,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# 2. Load Knowledge Base (ChromaDB)
print("📚 Loading Knowledge Base...")
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="business_Knowledge")

# 3. Enhanced System Prompt
# {lang_instruction} is filled at runtime based on detected language
SYSTEM_PROMPT = """You are the official Sales Executive for AGS Gadgets UAE.
Your goal is to answer customer questions professionally.

IMPORTANT: {lang_instruction}

RULES:
1. STRICTLY follow the language instruction above — never switch languages.
2. For prices: Only provide the exact price found in the CONTEXT. Do not say "To place an order" if the user only asked for a price.
3. Keep answers short and professional. DO NOT make up stories or talk about videos/galleries.
4. Only answer what is asked. Do not add unrelated information.
5. If the user asks an unrelated or out-of-context question (e.g., about cyber security, history, or products not in the CONTEXT), you MUST reply exactly with: "I don't know." or "Mujhe nahi maloom." do not try to guess.

CONTEXT:
{context}
"""

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

    # 1. Detect Language Early — individual tokens only (multi-word phrases won't match with 'in')
    # Match Urdu (Arabic) or Hindi (Devanagari) script
    URDU_SCRIPT_CHARS = re.compile(r'[\u0600-\u06FF\u0900-\u097F]')
    try:
        # Prevent false Urdu detection from english words like "order", "mobile", "price"
        roman_urdu_words = ["hai", "kya", "kitne", "batao", "salam", "chahiye",
                            "aoa", "kaise", "karna", "karo", "tareeqa", "keemat", "lena",
                            "mangwana", "mujhe", "nahi", "haan", "jee"]
        if URDU_SCRIPT_CHARS.search(user_message):
            user_lang = "ur"
        elif any(word in user_message_lower.split() for word in roman_urdu_words):
            user_lang = "ur"
        else:
            user_lang = detect(user_message)
    except:
        user_lang = "en"

    # --- GREETINGS LOGIC ---
    english_greetings = {"hi", "hello", "hey", "merhaba", "hiya", "howdy", "high"}
    urdu_greetings = {
        "salam", "salaam", "aoa", "asalam", "assalam", 
        "walaikum", "assalamoalaikum", "assalamu", "alaykum",
        "asalaam", "assalaam",
        "سلام", "السلام", "علیکم", "وعلیکم", "السلامعلیکم", "وعلیکماسلام",
        "नमस्ते", "नमस्कार", "हेलो", "सलाम" # Devanagari Hindi
    }

    # Extract clean words without punctuation
    words = re.sub(r'[^\w\s]', '', user_message_lower).split()
    
    is_english_greeting = False
    is_urdu_greeting = False

    # If the message is short (1 to 3 words) and contains a greeting word
    if 1 <= len(words) <= 3:
        for w in words:
            if w in urdu_greetings:
                is_urdu_greeting = True
                break
            elif w in english_greetings:
                is_english_greeting = True
                break

    if is_english_greeting or is_urdu_greeting:
        # Reset state if they greet again
        if sender in USER_STATES:
            del USER_STATES[sender]
            
        if is_urdu_greeting:
            return "وعلیکم اسلام! اے جی ایس گیجٹس یو ای کی خدمت میں خوش آمدید! میں آپکی کیا مدد کر سکتا ہوں؟", "ur"
        else:
            return "Hello! Welcome to AGS Gadgets UAE. How can I help you today?", "en"

    # --- ORDER FLOW STATE MACHINE ---
    if sender in USER_STATES:
        state = USER_STATES[sender]
        
        # Step 2: User provides details (Name, Number, Location) -> Ask what product
        if state["step"] == "awaiting_details":
            if len(user_message_lower) > 5:
                # Clean up "My name is", "location is", etc.
                clean_details = user_message
                
                # Replace "and" or "aur" with comma for better splitting
                clean_details = re.sub(r"(?i)\b\s+and\s+\b", ", ", clean_details)
                clean_details = re.sub(r"(?i)\b\s+aur\s+\b", ", ", clean_details)
                
                removal_patterns = [
                    r"(?i)my\s+name\s+is\s*", r"(?i)mera\s+naam\s+", r"(?i)i\s+am\s+", r"(?i)\bname\s*is\s*", r"(?i)\bname\s*:\s*",
                    r"(?i)location\s+is\s*", r"(?i)location\s*:\s*", r"(?i)living\s+in\s*", r"(?i)\bfrom\s*", r"(?i)meri\s+location\s*",
                    r"(?i)mobile\s+no\s+is\s*", r"(?i)mobile\s+number\s+is\s*", r"(?i)phone\s+number\s+is\s*",
                    r"(?i)\bnumber\s+is\s*", r"(?i)\bmobile\s*:\s*", r"(?i)\bphone\s*:\s*", r"(?i)\bcontact\s*:\s*",
                    r"(?i)\bhai\b", r"(?i)\bhoon\b"
                ]
                for pattern in removal_patterns:
                    clean_details = re.sub(pattern, "", clean_details)
                
                # Clean up spaces and commas, and apply Title Case
                clean_details = ", ".join([p.strip().title() for p in clean_details.split(",") if p.strip()])
                if not clean_details:
                    clean_details = user_message.strip()

                USER_STATES[sender]["step"] = "awaiting_product"
                USER_STATES[sender]["details"] = clean_details
                if user_lang == "ur":
                    return "شکریہ! آپ کیا آرڈر کرنا چاہتے ہیں؟ براہ کرم پروڈکٹ کا نام بتائیں (جیسے کہ سیمسنگ موبائل یا لیپ ٹاپ)۔", user_lang
                return "Thank you. What order would you like to place? (Please specify the product, like mobile or laptop)", user_lang
            else:
                if user_lang == "ur":
                    return "براہ کرم اپنی مکمل تفصیل دیں: پورا نام، موبائل نمبر، اور گوگل میپس لوکیشن۔", user_lang
                return "Please provide your complete details: Full Name, Mobile Number, and Google Maps Location to proceed.", user_lang

        # Step 3: User provides product -> Ask for confirmation
        elif state["step"] == "awaiting_product":
            if len(user_message_lower) > 2:
                # Extract purely the product name
                clean_product = user_message_lower
                phrases_to_remove = [
                    r"(?i)\bi\s+want\s+to\s+buy\b\s*", r"(?i)\bi\s+want\s+to\s+order\b\s*", r"(?i)\bi\s+would\s+like\s+to\s+order\b\s*",
                    r"(?i)\bi\s+would\s+like\b\s*", r"(?i)\bi\s+want\b\s*", r"(?i)\bi\s+need\b\s*", r"(?i)\bgive\s+me\b\s*", r"(?i)\bmujhe\b\s*",
                    r"(?i)\boff?er\b\s*", r"(?i)\bpurchase\b\s*", r"(?i)\bbuy\b\s*", r"(?i)\bget\b\s*", r"(?i)\bto\b\s*",
                    r"(?i)\border\b\s*", r"(?i)\ban?\b\s*", r"(?i)\bthe\b\s*", r"(?i)\bor\b\s*",
                    r"(?i)\s+chahiye\b", r"(?i)\s+dedo\b", r"(?i)\s+order\s+karna\s+hai\b", r"(?i)\s+lena\s+hai\b", 
                    r"(?i)\s+mangwana\s+hai\b", r"(?i)\bplease\b\s*", r"(?i)\s+please\b", r"(?i)\bi\s+am\s+looking\s+for\b\s*"
                ]
                for pattern in phrases_to_remove:
                    clean_product = re.sub(pattern, "", clean_product)
                
                # Title case for better formatting
                clean_product = clean_product.strip().title()
                # Fallback if empty after regex
                if not clean_product:
                    clean_product = user_message.strip()

                USER_STATES[sender]["step"] = "awaiting_confirmation"
                USER_STATES[sender]["product"] = clean_product
                if user_lang == "ur":
                    return f"آپکے آرڈر کی تفصیل:\nپروڈکٹ: {clean_product}\nتفصیل: {state['details']}\n\nآرڈر تصدیق کرنے کے لیے 'جی ہاں' لکھیں۔", user_lang
                return f"Your order details:\nProduct: {clean_product}\nDetails: {state['details']}\n\nPlease reply with 'Yes' to confirm your order.", user_lang
            else:
                if user_lang == "ur":
                    return "براہ کرم پروڈکٹ کا نام بتائیں جو آپ آرڈر کرنا چاہتے ہیں۔", user_lang
                return "Please specify the product you want to order.", user_lang

        # Step 4: User confirms -> Finish order
        elif state["step"] == "awaiting_confirmation":
            confirm_words = ["yes", "y", "confirm", "haan", "جی ہاں", "ہاں", "جی", "ٹھیک ہے", "تصدیق", "हाँ", "जी"]
            cancel_words = ["no", "cancel", "nahi", "نہیں", "نہی", "منسوخ", "नहीं", "कैंसिल"]
            if any(w in user_message_lower for w in confirm_words):
                CONFIRMED_ORDERS_QUEUE[sender] = {
                    "product": state.get("product", "Unknown Product"),
                    "details": state.get("details", "")
                }
                del USER_STATES[sender]
                if user_lang == "ur":
                    return "آپ کا آرڈر کنفرم ہو گیا ہے! آپ کا آرڈر کامیابی سے جمع کر دیا گیا ہے۔", user_lang
                return "Order Confirmed! Your order has been placed successfully.", user_lang
            elif any(w in user_message_lower for w in cancel_words):
                del USER_STATES[sender]
                if user_lang == "ur":
                    return "آرڈر منسوخ کر دیا گیا۔ میں آپکی مزید کیا مدد کر سکتا ہوں؟", user_lang
                return "Order cancelled. How else can I help you today?", user_lang
            else:
                if user_lang == "ur":
                    return "براہ کرم آرڈر تصدیق کرنے کے لیے 'جی ہاں' یا منسوخ کرنے کے لیے 'نہیں' لکھیں۔", user_lang
                return "Please reply with 'Yes' to confirm your order, or 'No' to cancel.", user_lang

    # --- INTENT DETECTION (If not in an active order flow) ---
    # Keyword-pair: works for Roman Urdu, Arabic-script Urdu, and English.
    ORDER_WORDS = [
        "order", "khareedna", "kharidna", "mangwana", "book", "purchase", "buy",
        "آرڈر", "خرید", "منگوا",
        "ऑर्डर", "बुक", "मंगवा", "खरीद",
        "اوڈر", "آڈر", "اودر", "oder", "auder",
    ]
    INTENT_WORDS = [
        "kaise", "karna", "karo", "tareeqa", "process", "place", "chahiye",
        "dena", "how", "want", "would", "kia", "kya", "procedure", "steps",
        "ha", "hain", "کیسے", "کرنا", "طریقہ", "کریں", "دینا",
        "कैसे", "करना", "तरीका", "चाहिए", "क्या", "हैं", "है",
    ]
    EXACT_ORDER_PHRASES = [
        "process of placing an order", "how to place an order", "how to order",
        "place an order", "process of order", "order process", "want to order",
        "آرڈر کیسے کریں", "آرڈر کرنا ہے",
        "آرڈر دینے کا طریقہ", "آرڈر پلیس",
        "آرڈر پلیس کرنے کے لیے کیا کرنا چاہیے؟",
        "آرڈر دینے کے لیے مجھے کیا کرنے کی ضرورت ہے؟",
    ]

    has_order  = any(w in user_message_lower for w in ORDER_WORDS)
    has_intent = any(w in user_message_lower for w in INTENT_WORDS)
    has_exact  = any(ph in user_message_lower for ph in EXACT_ORDER_PHRASES)

    if has_exact or (has_order and has_intent):
        USER_STATES[sender] = {"step": "awaiting_details"}
        if user_lang == "ur":
            return "آرڈر دینے کے لیے براہ کرم اپنا پورا نام، موبائل نمبر، اور گوگل میپس لوکیشن بتائیں۔", user_lang
        return "To place an order, please share your Full Name, Mobile Number, and Google Maps Location.", user_lang

    # 2. Get Context (FIXED: replaced undefined top_k with 3)
    context = retrieve_context(user_message, top_k=3)
    
    # 3. Fallback Context if DB is empty
    if not context or len(context.strip()) < 10:
        if user_lang == "ur":
            context = "AGS Gadgets UAE phones bechta hai. S25 Ultra 210 AED ka hai. Cash on delivery pure UAE mein hai. Order ke liye Name, Number aur Location chahiye."
        else:
            context = "AGS Gadgets UAE sells premium phones. S25 Ultra is 210 AED. Cash on delivery available. To order, we need: Name, Number, and Location."

    # 4. Build strict Prompt — inject language instruction explicitly
    if user_lang == "ur":
        lang_instruction = "Reply ONLY in Urdu (Roman Urdu is also acceptable). Do NOT reply in English."
    else:
        lang_instruction = "Reply ONLY in English."

    full_prompt = f"{SYSTEM_PROMPT.format(context=context, lang_instruction=lang_instruction)}\n\nUser: {user_message}\nAssistant:"

    # 5. Generate — no min_new_tokens so model stops naturally
    response = generator(
        full_prompt,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.3,          # Lower = more factual, less hallucination
        top_p=0.85,
        top_k=40,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    # 6. Extract the Assistant's reply only
    generated_text = response[0]["generated_text"]

    if "Assistant:" in generated_text:
        answer = generated_text.split("Assistant:")[-1].strip()
    else:
        answer = generated_text.replace(full_prompt, "").strip()

    # 7. Aggressively strip hallucinated fake-conversation content
    # Stop at any role label or separator the model might invent
    STOP_PATTERNS = [
        "User:", "Customer:", "Assistant:",
        "Video Response:", "Human:", "Bot:",
        "\n\n",   # double newline = model starting a new topic
    ]
    for pattern in STOP_PATTERNS:
        if pattern in answer:
            answer = answer.split(pattern)[0].strip()

    # Strip lines that are pure emoji / noise BUT keep Urdu/Arabic-script lines
    clean_lines = []
    for line in answer.splitlines():
        stripped = line.strip()
        # Keep line if it has >= 3 real characters (any Unicode letter counts)
        if len(re.findall(r'[\w\u0600-\u06FF]', stripped)) >= 3:
            clean_lines.append(stripped)
    answer = " ".join(clean_lines).strip()

    # 8. Check for "I don't know" from the LLM based on our strict prompt rule
    dont_know_variants = ["i don't know", "mujhe nahi maloom", "i do not know", "i am not sure", "مجھے نہیں معلوم", "معلوم نہیں", "मुझे नहीं पता", "मुझे नहीं मालूम"]
    is_dont_know = any(variant in answer.lower() for variant in dont_know_variants)
    
    # Check for price queries specifically
    price_keywords_en = ["price", "cost", "how much"]
    price_keywords_ur = ["kitne", "keemat", "price", "rate", "کتنے", "قیمت", "ریٹ", "कितने", "कीमत", "रेट"]
    is_price_query = (
        any(kw in user_message.lower() for kw in price_keywords_en) or
        any(kw in user_message.lower() for kw in price_keywords_ur)
    )

    if is_dont_know:
         if user_lang == "ur":
             return "معذرت، اس بارے میں معلومات دستیاب نہیں۔", user_lang
         else:
             return "I don't know.", user_lang

    # 9. Generic fallback for empty / garbage answer
    if len(answer) < 5:
        if is_price_query:
            if user_lang == "ur":
                return "معذرت، اس پروڈکٹ کی قیمت تفصیل میں دستیاب نہیں۔ مزید معلومات کے لیے 0563235075 پر رابطہ کریں۔", user_lang
            return "I'm sorry, I don't have the price for that specific item. Please contact admin.", user_lang

        if user_lang == "ur":
             return "معذرت، اس بارے میں معلومات دستیاب نہیں۔ براہ کرم 0563235075 پر کال کریں۔", user_lang
        return "Sorry, I don't have enough information on that. You can also reach us at 0563235075.", user_lang

    return answer, user_lang