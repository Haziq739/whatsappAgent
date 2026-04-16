import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import chromadb
import config
from langdetect import detect
import re
from deep_translator import GoogleTranslator

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

# 3. Enhanced System Prompt
# {lang_instruction} is filled at runtime based on detected language
SYSTEM_PROMPT = """CONTEXT:
{context}

You are the official Sales Executive for AGS Gadgets UAE.
Your goal is to answer customer questions professionally and STRICTLY based on the CONTEXT provided above.

IMPORTANT: {lang_instruction}

STRICT RULES:
1. ONLY use information from the CONTEXT above. Do NOT use outside knowledge.
2. If the user asks for available products, strictly list the models found in the CONTEXT.
3. For prices: Only provide the exact price found in the CONTEXT for that specific model.
4. If the information is not in the CONTEXT, you must say: "Let me confirm this for you. Please contact us on WhatsApp or call 0563235075."
5. Keep answers short, professional, and sales-oriented.
6. Do NOT hallucinate names, prices, or products.
7. NEVER reply in Hindi script or Devanagari characters. Only use English, Roman Urdu, or Urdu script as instructed.
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
            user_lang = "ru"   # Roman Urdu — distinct from both "ur" and "en"
        elif re.search(r'[\u0900-\u097F]', user_message):
            # If Hindi characters are found (Whisper mistranscription), default to Roman Urdu for response
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
            return "وعلیکم اسلام! اے جی ایس گیجٹس یو ای کی خدمت میں خوش آمدید! میں آپکی کیا مدد کر سکتا ہوں?", "ur"
        else:
            return "Hello! Welcome to AGS Gadgets UAE. How can I help you today?", "en"

    # --- ORDER FLOW STATE MACHINE ---
    if sender in USER_STATES:
        state = USER_STATES[sender]
        
        # Step 2: User provides details (Name, Number, Location) -> Ask what product
        if state["step"] == "awaiting_details":
            # Validate that the input looks like actual contact info (must have a digit for phone number)
            has_digit = bool(re.search(r'\d', user_message))
            has_alpha = bool(re.search(r'[a-zA-Z\u0600-\u06FF]{2,}', user_message))
            is_valid_details = has_digit and has_alpha and len(user_message_lower) > 5
            
            if is_valid_details:
                # Clean up "My name is", "location is", etc.
                clean_details = user_message
                
                # Replace "and" or "aur" or "اور" with comma for better splitting
                clean_details = re.sub(r"(?i)\b\s+and\s+\b", ", ", clean_details)
                clean_details = re.sub(r"(?i)\b\s+aur\s+\b", ", ", clean_details)
                clean_details = re.sub(r"\s+اور\s+", ", ", clean_details)
                
                removal_patterns = [
                    r"(?i)my\s+name\s+is\s*", r"(?i)mera\s+naam\s+", r"(?i)i\s+am\s+", r"(?i)\bname\s*is\s*", r"(?i)\bname\s*:\s*",
                    r"(?i)location\s+is\s*", r"(?i)location\s*:\s*", r"(?i)living\s+in\s*", r"(?i)\bfrom\s*", r"(?i)meri\s+location\s*",
                    r"(?i)mobile\s+no\s+is\s*", r"(?i)mobile\s+number\s+is\s*", r"(?i)phone\s+number\s+is\s*",
                    r"(?i)\bnumber\s+is\s*", r"(?i)\bmobile\s*:\s*", r"(?i)\bphone\s*:\s*", r"(?i)\bcontact\s*:\s*",
                    r"(?i)\bhai\b", r"(?i)\bhoon\b",
                    # Stricter standalone removal for Name cleanup
                    r"(?i)\bmobile\s+number\b", r"(?i)\bphone\s+number\b", r"(?i)\bmobile\s+no\b", r"(?i)\bmobile\b", r"(?i)\bphone\b", r"(?i)\bnumber\b", r"(?i)\blocation\b", r"(?i)\baddress\b",
                    # Urdu Script patterns
                    r"میرا\s+نام\s+", r"نام\s+ہے\s*", r"لوکیشن\s+ہے\s*", r"لوکیشن\s*:\s*", r"لوکیشن\b", r"رہتا\s+ہوں\s*", 
                    r"موبائل\s+نمبر\s+", r"موبائل\b", r"نمبر\b", r"نمبر\s+ہے\s*", r"\s+ہے\b"
                ]
                for pattern in removal_patterns:
                    clean_details = re.sub(pattern, "", clean_details)
                
                # Clean up spaces and commas, and apply Title Case
                clean_details = ", ".join([p.strip().title() for p in clean_details.split(",") if p.strip()])
                
                # --- NEW: TRANSLATE TO ENGLISH ---
                try:
                    # Translate to English so confirmation message shows English details
                    if any(ord(c) > 127 for c in clean_details):
                        clean_details = GoogleTranslator(source='auto', target='en').translate(clean_details)
                except Exception as e:
                    print(f"Translation error in details: {e}")

                # --- NEW: DEDUPLICATION (Fixes Whisper repetition bug) ---
                # Split into pieces, remove duplicates, and rejoin
                pieces = [p.strip() for p in clean_details.split(",") if p.strip()]
                unique_pieces = []
                for p in pieces:
                    if p not in unique_pieces:
                        unique_pieces.append(p)
                clean_details = ", ".join(unique_pieces)
                
                # Final check for repeating words within pieces (Surgical deduplication)
                words = clean_details.split()
                if len(words) > 10: # Only deduplicate if reasonably long
                    clean_words = []
                    for i, w in enumerate(words):
                        if i == 0 or w != words[i-1]:
                            clean_words.append(w)
                    clean_details = " ".join(clean_words)

                if not clean_details:
                    clean_details = user_message.strip()

                USER_STATES[sender]["step"] = "awaiting_product"
                USER_STATES[sender]["details"] = clean_details
                if user_lang == "ur":
                    return "شکریہ! آپ کیا آرڈر کرنا چاہتے ہیں؟ براہ کرم پروڈکٹ کا نام بتائیں (جیسے کہ سیمسنگ موبائل یا لیپ ٹاپ)۔", user_lang
                elif user_lang == "ru":
                    return "Shukriya! Aap kya order karna chahte hain? Barah e meherbani product ka naam batayein (jaise Samsung mobile ya laptop).", user_lang
                return "Thank you. What order would you like to place? (Please specify the product, like mobile or laptop)", user_lang
            else:
                if user_lang == "ur":
                    return "براہ کرم اپنی مکمل تفصیل دیں: پورا نام، موبائل نمبر، اور گوگل میپس لوکیشن۔", user_lang
                elif user_lang == "ru":
                    return "Barah meherbani apni mukammal details dein: Poora Naam, Mobile Number, aur Google Maps Location.", user_lang
                return "Please provide your complete details: Full Name, Mobile Number, and Google Maps Location to proceed.", user_lang

        # Step 3: User provides product -> Ask for confirmation
        elif state["step"] == "awaiting_product":
            if len(user_message_lower) > 2:
                # Extract purely the product name
                clean_product = user_message_lower
                phrases_to_remove = [
                    r"(?i)\bi\s+want\s+to\s+buy\b\s*", r"(?i)\bi\s+want\s+to\s+order\b\s*", r"(?i)\bi\s+would\s+like\s+to\s+order\b\s*",
                    r"(?i)\bi\s+would\s+like\b\s*", r"(?i)\bi\s+want\b\s*", r"(?i)\bi\s+need\b\s*", r"(?i)\bgive\s+me\b\s*",
                    r"(?i)\bmujh[yeai]*\b\s*", r"(?i)\bmugh[yeai]*\b\s*", r"(?i)\bmujhye\b\s*", # Mughy, Mujhye, Mujhe, Mujhay, Muje
                    r"(?i)\boff?er\b\s*", r"(?i)\bpurchase\b\s*", r"(?i)\bbuy\b\s*", r"(?i)\bget\b\s*", r"(?i)\bto\b\s*",
                    r"(?i)\border\b\s*", r"(?i)\ban?\b\s*", r"(?i)\bthe\b\s*", r"(?i)\bor\b\s*",
                    # Roman Urdu Fillers (Surgical removal)
                    r"(?i)\bcha[hyei]+[yei]*\b", r"(?i)\bchhaye\b", r"(?i)\bchahiye\b", r"(?i)\bchahia\b",
                    r"(?i)\bdedo\b", r"(?i)\border\s+karna\s+hai\b", r"(?i)\blena\s+hai\b", r"(?i)\bleina\b",
                    r"(?i)\bmangwana\s+hai\b", r"(?i)\bmangwana\b", r"(?i)\bbook\s+karna\s+hai\b", r"(?i)\bbook\s+karna\b", r"(?i)\bbook\s+krna\b",
                    r"(?i)\bplease\b", r"(?i)\bi\s+am\s+looking\s+for\b\s*",
                    r"(?i)\bh\b\s*", r"(?i)\bhai\b\s*", r"(?i)\bhain\b\s*", r"(?i)\bn\b\s*$",
                    # Urdu Script patterns
                    r"چاہیے", r"چاہئیے", r"منگوانا\s+ہے", r"خریدنا\s+ہے", r"آڈر\s+کرنا\s+ہے"
                ]
                for pattern in phrases_to_remove:
                    clean_product = re.sub(pattern, "", clean_product)
                
                # Final strip of any non-alphanumeric noise at start/end
                clean_product = re.sub(r"^[^\w]+|[^\w]+$", "", clean_product).strip().title()
                
                # Fallback if empty
                if not clean_product:
                    clean_product = user_message.strip()

                USER_STATES[sender]["step"] = "awaiting_confirmation"
                USER_STATES[sender]["product"] = clean_product
                if user_lang == "ur":
                    return f"آپ کے آرڈر کی تفصیل یہ ہے:\nپروڈکٹ: {clean_product}\nتفصیل: {state['details']}\n\nآرڈر کنفرم کرنے کے لیے 'جی ہاں' لکھیں۔ شکریہ!", user_lang
                elif user_lang == "ru":
                    return f"Aap ke order ki detail yeh hai:\nProduct: {clean_product}\nDetails: {state['details']}\n\nOrder confirm karne ke liye barah e meherbani 'Haan' likhein. Shukriya!", user_lang
                return f"Here are your order details:\nProduct: {clean_product}\nDetails: {state['details']}\n\nPlease reply with 'Yes' to confirm your order. Thank you!", user_lang
            else:
                if user_lang == "ur":
                    return "براہ کرم پروڈکٹ کا نام بتائیں جو آپ آرڈر کرنا چاہتے ہیں۔", user_lang
                elif user_lang == "ru":
                    return "Barah meherbani product ka naam batayein jo aap order karna chahte hain.", user_lang
                return "Please specify the product you want to order.", user_lang

        # Step 4: User confirms -> Finish order
        elif state["step"] == "awaiting_confirmation":
            confirm_words = ["yes", "y", "confirm", "haan", "ha", "ji", "theek", "ok",
                             "جی ہاں", "ہاں", "جی", "ٹھیک ہے", "تصدیق"]
            cancel_words = ["no", "cancel", "nahi", "nhi", "na",
                            "نہیں", "نہی", "منسوخ"]
            if any(w in user_message_lower for w in confirm_words):
                CONFIRMED_ORDERS_QUEUE[sender] = {
                    "product": state.get("product", "Unknown Product"),
                    "details": state.get("details", "")
                }
                del USER_STATES[sender]
                if user_lang == "ur":
                    return "آپ کا آرڈر کنفرم ہو گیا ہے! آپ کا آرڈر کامیابی سے جمع کر دیا گیا ہے۔", user_lang
                elif user_lang == "ru":
                    return "Aap ka order confirm ho gaya! Aap ka order kamyabi se submit kar diya gaya hai.", user_lang
                return "Order Confirmed! Your order has been placed successfully.", user_lang
            elif any(w in user_message_lower for w in cancel_words):
                del USER_STATES[sender]
                if user_lang == "ur":
                    return "آرڈر منسوخ کر دیا گیا۔ میں آپکی مزید کیا مدد کر سکتا ہوں؟", user_lang
                elif user_lang == "ru":
                    return "Order cancel kar diya gaya. Main aap ki aur kya madad kar sakta hoon?", user_lang
                return "Order cancelled. How else can I help you today?", user_lang
            else:
                if user_lang == "ur":
                    return "براہ کرم آرڈر تصدیق کرنے کے لیے 'جی ہاں' یا منسوخ کرنے کے لیے 'نہیں' لکھیں۔", user_lang
                elif user_lang == "ru":
                    return "Barah meherbani order confirm karne ke liye 'Haan' ya cancel karne ke liye 'Nahi' likhein.", user_lang
                return "Please reply with 'Yes' to confirm your order, or 'No' to cancel.", user_lang

    # --- INTENT DETECTION (If not in an active order flow) ---
    # Keyword-pair: works for Roman Urdu, Arabic-script Urdu, and English.
    ORDER_WORDS = [
        "order", "orderz", "orders", "khareedna", "kharidna", "mangwana", "book", "purchase", "buy",
        "آرڈر", "خرید", "منگوا", "اوڈر", "آڈر", "اودر", "oder", "auder", "audar", "audder",
    ]
    INTENT_WORDS = [
        "kaise", "kaisay", "kaunon", "karna", "karo", "tareeqa", "target", "tareeka", "tarika", "tareekah", 
        "process", "place", "chahiye", "chhaye", "chaye", "chahia",
        "dena", "denay", "dainay", "denae", "how", "want", "would", "procedure", "steps", "lena", "leina", "mangwana",
        "ha", "hain", "hai", "hey", "hy", "kya", "kia", "h",
        "کیسے", "کرنا", "طریقہ", "کریں", "دینا", "\u0686\u0627\u06c1\u06cc\u06d2", "کیا", "ہے", "ہیں",
    ]
    EXACT_ORDER_PHRASES = [
        "process of placing an order", "how to place an order", "how to order",
        "place an order", "process of order", "order process", "want to order",
        "order karna hai", "order krna hai", "order krna h", "muje order karna hai",
        "mujhe order karna hai", "order dene ka tareeqa", "order kaise karein",
        "order dene ka tareeqa kya hai", "order dene ka tarika", "order karne ka tareeqa",
        "order dene ka tareeka kia hai", "order karne ka tarika", "order dene ka tereeqa",
        "order denay ka tareeka", "order dainay ka tareeqa", "order denay ka tarika",
        "آرڈر کیسے کریں", "آرڈر کرنا ہے", "آرڈر دینے کا طریقہ", "آرڈر پلیس",
        "آرڈر پلیس کرنے کے لیے کیا کرنا چاہیے؟",
        "آرڈر دینے کے لیے مجھے کیا کرنے کی ضرورت ہے؟",
        "آرڈر دینے کا طریقہ کیا ہے", "آرڈر کرنے کا طریقہ", "آرڈر دینے کا طریقہ کیا ہے؟",
    ]

    # Clean words for intent detection (strip punctuation)
    clean_msg_for_intent = re.sub(r'[^\w\s\u0600-\u06FF]', '', user_message_lower)
    words_for_intent = clean_msg_for_intent.split()

    has_order  = any(w in words_for_intent for w in ORDER_WORDS)
    has_intent = any(w in words_for_intent for w in INTENT_WORDS)
    has_exact  = any(ph in clean_msg_for_intent for ph in EXACT_ORDER_PHRASES)

    PRODUCT_CAT_WORDS = ["samsung", "iphone", "pixel", "redmi", "laptop", "mobile", "phone", "ultra", "pro", "max"]
    has_product = any(w in words_for_intent for w in PRODUCT_CAT_WORDS)

    if has_exact or (has_order and has_intent) or (has_product and has_intent):
        USER_STATES[sender] = {"step": "awaiting_details"}
        if user_lang == "ur":
            return "آرڈر دینے کے لیے براہ کرم اپنا پورا نام، موبائل نمبر، اور گوگل میپس لوکیشن بتائیں۔", user_lang
        elif user_lang == "ru":
            return "Order dene ke liye barah meherbani apna poora naam, mobile number, aur Google Maps location batayein.", user_lang
        return "To place an order, please share your Full Name, Mobile Number, and Google Maps Location.", user_lang

    # 2. Get Context (Increased top_k to 15 for better coverage of multiple products)
    context = retrieve_context(user_message, top_k=15)
    
    # 3. Fallback Context if DB is empty
    if not context or len(context.strip()) < 10:
        if user_lang == "ur":
            context = "AGS Gadgets UAE phones bechta hai. S25 Ultra 210 AED ka hai. Cash on delivery pure UAE mein hai. Order ke liye Name, Number aur Location chahiye."
        else:
            context = "AGS Gadgets UAE sells premium phones. S25 Ultra is 210 AED. Cash on delivery available. To order, we need: Name, Number, and Location."

    # 4. Build strict Prompt — inject language instruction explicitly
    if user_lang == "ur":
        lang_instruction = "Reply ONLY in Urdu script (Arabic letters). Do NOT reply in English or Roman Urdu."
    elif user_lang == "ru":
        lang_instruction = "Reply ONLY in Roman Urdu (e.g. 'price kya hai', 'mil jayega'). Do NOT reply in English or Urdu script."
    else:
        lang_instruction = "Reply ONLY in English."

    full_prompt = f"{SYSTEM_PROMPT.format(context=context, lang_instruction=lang_instruction)}\n\nUser: {user_message}\nAssistant:"

    # 5. Generate — no min_new_tokens so model stops naturally
    response = generator(
        full_prompt,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.01,         # Extremely low = strictly factual
        top_p=0.85,
        top_k=40,
        repetition_penalty=1.1,   # Lower = helps numerical answers (prices)
        no_repeat_ngram_size=3,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    # 6. Extract the Assistant's reply only — Since we use return_full_text=False, 
    # the entire generated_text is the answer itself.
    answer = response[0]["generated_text"].strip()

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

    with open("debug_log.txt", "a", encoding="utf-8") as f:
        f.write(f"\nFinal Answer (After Cleanup):\n{answer}\n-----------------\n")

    # 8. Detect and filter "trash" output (hallucinations like "ARY OL", "Original", "Yes", etc.)
    gadget_keywords = ["price", "aed", "order", "delivery", "shipping", "product", "available",
                       "phone", "mobile", "ultra", "samsung", "redmi", "warranty", "cod",
                       "cash", "day", "box", "seal", "charger", "contact", "whatsapp",
                       "درہم", "قیمت", "آرڈر", "ڈیلیوری", "فون", "دستیاب", "وارنٹی"]
    has_gadget_info = any(kw in answer.lower() for kw in gadget_keywords)
    answer_word_count = len(answer.split())
    
    # IMPROVED TRASH FILTER: Rejects non-business hallucinations
    is_trash = (
        re.match(r'^[A-Z\s]{2,10}$', answer)           # All-caps gibberish
        or len(answer) < 2                              # Empty/tiny
        or (answer_word_count <= 2 and not has_gadget_info)  # 1-2 random words
        # Rejects stuff like "The person who built it" if it has no gadget context
        or (not has_gadget_info and not any(w in answer.lower() for w in ["sorry", "maafi", "information", "available"]))
    )
    if is_trash:
        answer = "" # Trigger fallback logic below
    dont_know_variants = ["i don't know", "mujhe nahi maloom", "i do not know", "i am not sure", "مجھے نہیں معلوم", "معلوم نہیں"]
    is_dont_know = any(variant in answer.lower() for variant in dont_know_variants)
    
    # Check for price queries specifically — must also mention a gadget-related term
    price_keywords = ["price", "cost", "how much", "kitne", "keemat", "rate", "کتنے", "قیمت", "ریٹ"]
    gadget_context_words = ["phone", "mobile", "samsung", "redmi", "iphone", "pixel", "laptop",
                            "macbook", "ultra", "pro", "max", "s25", "s24", "s23", "14c",
                            "a15", "a25", "a35", "a55", "fold", "flip", "hp", "dell",
                            "lenovo", "asus", "rog", "thinkpad", "elitebook",
                            "فون", "موبائل", "سیمسنگ", "ریڈمی", "آئی فون", "لیپ ٹاپ"]
    has_price_kw = any(kw in user_message.lower() for kw in price_keywords)
    has_gadget_context = any(gw in user_message.lower() for gw in gadget_context_words)
    is_price_query = has_price_kw and has_gadget_context

    # Check for catalog/list keywords — STRICT: must mention products/phones/stock/available/laptop
    catalog_business_words = ["products", "models", "stock", "phones", "phone",
                              "mobile", "mobiles", "laptop", "laptops", "available",
                              "فون", "موبائل", "موبائلز", "پروڈکٹ", "دستیاب"]
    is_catalog_query = any(kw in user_message.lower() for kw in catalog_business_words)

    # --- 9. DIRECT RESPONSES (Catalog & Price) ---
    
    # 9a. PRICE QUERY takes precedence
    if is_price_query:
        # Price logic extracted into the block below to allow RAG context searching
        pass 

    # 9b. CATALOG/LIST (Inventory check)
    elif is_catalog_query:
        if user_lang == "ur":
            return "ہمارے پاس دستیاب موبائلز یہ ہیں: سیمسنگ ایس 25 الٹرا، ایس 24 الٹرا (ہائی کوالٹی کاپی)، ایس 23 الٹرا، اور ریڈمی 14C۔ اس کے علاوہ آئی فون، گوگل پکسل، اور لیپ ٹاپس بھی دستیاب ہیں۔", user_lang
        elif user_lang == "ru":
            return "Humare pas available products ye hain: Samsung S25 Ultra, S24 Ultra (Copy), S23 Ultra, Redmi 14C. Iske alawa iPhone, Google Pixel, aur Laptops bhi available hain.", user_lang
        return "The products currently available at AGS Gadgets UAE include: Samsung Galaxy S25 Ultra, S24 Ultra (Quality Copy), S23 Ultra, Redmi 14c, iPhones, Google Pixel, and Laptops. Ask for specific model prices.", user_lang

    # 10. ROBUST PRICE & CATALOG SAFETY NET
    # If the LLM response is failing (short or "don't know"), 
    # we manually extract from the context.
    
    if (len(answer) < 5 or is_dont_know or is_price_query):
        # 10a. Price Safety Net (Manual Extraction)
        if is_price_query:
            # Extract main product terms from user query (e.g. S25, Redmi, Ultra)
            # Include numbers >= 2 digits (e.g. 25, 24, 14, 23)
            clean_msg = re.sub(r'[^\w\s]', '', user_message.lower())
            ignore_words = ['price', 'cost', 'kitne', 'keemat', 'hai', 'kia', 'kya', 'how', 'much']
            query_terms = [t for t in clean_msg.split() if (len(t) > 2 or t.isdigit()) and t not in ignore_words]
            
            context_price = ""
            best_line = ""
            max_matches = 0
            
            # Find a line that has MOST query term matches AND contains a price pattern
            for line in context.split("\n"):
                if not line.strip(): continue
                line_lower = line.lower()
                has_price = bool(re.search(r'(\d+\s*AED|\d+\s*درہم|price\s+is\s+\d+)', line, re.IGNORECASE))
                if has_price:
                    # Score and check for MUST-HAVE numbers or model IDs (e.g. 's25', '14c')
                    matches = 0
                    mandatory_fail = False
                    for term in query_terms:
                        if term in line_lower:
                            # Higher weight for specific model IDs
                            matches += 2 if any(c.isdigit() for c in term) else 1
                        elif any(c.isdigit() for c in term) and len(term) >= 2:
                            # If a specific number/model like 's25' is in query but NOT in this line
                            mandatory_fail = True
                    
                    # If it's a specific model number query, priority goes to the line with that number
                    current_score = matches if not mandatory_fail else 0.1
                    
                    if current_score > max_matches:
                        max_matches = current_score
                        best_line = line
            
            if best_line:
                # Search for digits + AED or "price is X" patterns in the confirmed line
                price_patterns = [
                    r"(\d+\s*AED)", r"(price\s+is\s+\d+)", r"(\d+\s*درہم)"
                ]
                for pattern in price_patterns:
                    matches = re.findall(pattern, best_line, re.IGNORECASE)
                    if matches:
                        context_price = matches[-1] # Take the last match if multiple
                        break
            
            if context_price:
                if user_lang == "ur":
                    return f"اس پروڈکٹ کی قیمت {context_price} ہے۔", user_lang
                elif user_lang == "ru":
                    return f"Is product ki price {context_price} hai.", user_lang
                return f"The price for this product is {context_price}.", user_lang

    # 10. Check if it's still a don't know response or empty/garbage answer
    if is_dont_know or len(answer) < 5:
        if user_lang == "ur":
            return "معذرت، اس بارے میں میرے پاس کوئی معلومات نہیں ہیں۔", user_lang
        elif user_lang == "ru":
            return "Maafi, is baare mein mere paas koi information nahi hai.", user_lang
        return "Sorry, I don't have any information about that.", user_lang

    return answer, user_lang