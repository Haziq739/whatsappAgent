import re
from langdetect import detect
from deep_translator import GoogleTranslator

# Tools integration
from tools.product_tool import product_lookup_tool
from tools.faq_tool import faq_lookup_tool
from tools.order_tool import order_creation_tool
from utils.firebase_db import get_recent_history
from utils.vector_memory import query_long_term_memory

USER_STATES = {}
CONFIRMED_ORDERS_QUEUE = {}

def _extract_customer_details(message):
    """
    Extract customer name, phone number, and location from a freeform message.
    Returns dict with keys: name, phone, location
    """
    text = message.strip()
    
    # --- Extract phone number ---
    phone = ""
    phone_match = re.search(r'(\+?\d[\d\s\-]{7,15}\d)', text)
    if phone_match:
        phone = re.sub(r'[\s\-]', '', phone_match.group(1))
    
    # --- Remove phone from text so it doesn't interfere with name/location ---
    text_no_phone = text
    if phone_match:
        text_no_phone = text[:phone_match.start()] + text[phone_match.end():]
    
    # --- Remove common filler words/phrases ---
    filler_patterns = [
        r'(?i)\bmy\s+name\s+is\b', r'(?i)\bmy\s+name\b', r'(?i)\bname\s+is\b',
        r'(?i)\bmobile\s*(no|number|num|#)?\s*(is)?\b',
        r'(?i)\bphone\s*(no|number|num|#)?\s*(is)?\b',
        r'(?i)\bcontact\s*(no|number|num|#)?\s*(is)?\b',
        r'(?i)\blocation\s*(is)?\b', r'(?i)\baddress\s*(is)?\b',
        r'(?i)\bgoogle\s*maps?\b',
        r'(?i)\bmera\s+naam\b', r'(?i)\bnaam\b',
        r'(?i)\bnumber\b', r'(?i)\bmobile\b', r'(?i)\bphone\b',
        r'(?i)\band\b', r'(?i)\baur\b',
        r'(?i)\bis\b',
    ]
    cleaned = text_no_phone
    for pat in filler_patterns:
        cleaned = re.sub(pat, ' ', cleaned)
    
    # Remove extra whitespace and punctuation-only fragments
    cleaned = re.sub(r'[,;:\-]+', ' , ', cleaned)  # normalize separators
    parts = [p.strip().strip(',').strip() for p in cleaned.split(',')]
    parts = [p for p in parts if p and len(p) > 1]
    
    # If we couldn't split, try splitting the cleaned text by multiple spaces
    if len(parts) <= 1:
        cleaned_flat = re.sub(r'\s{2,}', ' | ', cleaned.strip())
        parts = [p.strip() for p in cleaned_flat.split('|')]
        parts = [p for p in parts if p and len(p) > 1]
    
    name = ""
    location = ""
    
    # Heuristic: first text-only part is the name, last text-only part is the location
    text_parts = []
    for p in parts:
        # Remove any remaining digits-only parts
        p_clean = re.sub(r'\d+', '', p).strip()
        if p_clean and len(p_clean) > 1:
            text_parts.append(p_clean)
    
    if len(text_parts) >= 2:
        name = text_parts[0].strip().title()
        location = text_parts[-1].strip().title()
    elif len(text_parts) == 1:
        # Only one text part — try to guess if it's name or location
        name = text_parts[0].strip().title()
        location = "Not provided"
    
    # Clean up name — remove non-alpha chars
    name = re.sub(r'[^a-zA-Z\u0600-\u06FF\u0900-\u097F\s]', '', name).strip().title()
    if not name:
        name = "Customer"
    if not location:
        location = "Not provided"
    if not phone:
        phone = "Not provided"
    
    return {"name": name, "phone": phone, "location": location}

def generate_rag_response(user_message, sender="default"):
    """
    Main function called by app.py. Master intent router using the Tool system.
    """
    user_message_lower = user_message.lower().strip()
    
    # 0. FETCH RECENT CONVERSATION (Short-Term) & RELEVANT PAST (Long-Term)
    session_history = get_recent_history(sender, current_query=user_message, limit=5)
    long_term_context = query_long_term_memory(sender, user_message, limit=3)

    # --- 1. DETECT LANGUAGE ---
    URDU_SCRIPT_CHARS = re.compile(r'[\u0600-\u06FF]')
    roman_urdu_words = [
        "hai", "kya", "kitne", "batao", "salam", "chahiye", "chhaye", "chaye", "mughy", "mujhe",
        "aoa", "kaise", "karna", "karo", "tareeqa", "keemat", "lena", "leina", "shukriya",
        "mangwana", "mangwani", "nahi", "nhi", "haan", "jee", "btao", "bhai", "kal", "abhi",
        "milega", "kitna", "kahan", "aur", "ya", "muje", "mujhy", "mujhye", "se", "pe", "par",
        "mein", "ka", "ki", "ke", "ko", "na", "hua", "hogi", "hota", "aaye", "sakta", "sakte",
        "karein", "krein", "karne", "krne", "bhejo", "bhejein", "krna", "ho", "gaya", "ha", "h"
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

    # --- 2. GREETINGS ---
    english_greetings = {"hi", "hello", "hey", "merhaba", "hiya", "howdy"}
    urdu_greetings = {"salam", "salaam", "aoa", "asalam", "assalam", "walaikum", "سلام", "السلام", "علیکم", "وعلیکم"}
    words = re.sub(r'[^\w\s]', '', user_message_lower).split()
    
    if 1 <= len(words) <= 3:
        if any(w in urdu_greetings for w in words):
            if sender in USER_STATES: del USER_STATES[sender]
            return "وعلیکم اسلام! اے جی ایس گیجٹس یو ای کی خدمت میں خوش آمدید! میں آپکی کیا مدد کر سکتا ہوں?", "ur"
        elif any(w in english_greetings for w in words):
            if sender in USER_STATES: del USER_STATES[sender]
            return "Hello! Welcome to AGS Gadgets UAE. How can I help you today?", "en"

    # --- 3. INTENT PARSING (IMPROVED) ---
    ORDER_WORDS = ["order", "khareedna", "kharidna", "mangwana", "book", "purchase", "buy", "آرڈر", "خرید", "منگوا", "اوڈر"]
    INTENT_WORDS = ["kaise", "karna", "karo", "tareeqa", "process", "place", "chahiye", "how", "want", "lena", "mangwana"]

    # ---- PRODUCT brand/category words (these indicate a REAL product query) ----
    PRODUCT_CAT_WORDS = [
        "samsung", "iphone", "pixel", "redmi", "ultra", "pro", "macbook", "hp", "dell",
        "lenovo", "asus", "poco", "xiaomi", "nokia", "galaxy", "note", "fold", "flip",
        "سیمسنگ", "ریڈمی", "آئی فون"
    ]

    # ---- PRICE trigger words ----
    PRICE_WORDS_SINGLE = ["price", "cost", "kitne", "keemat", "rate", "کتنے", "قیمت", "ریٹ"]
    PRICE_PHRASES = ["how much"]

    # ---- FAQ keywords (COMPREHENSIVE — covers delivery, COD, warranty, returns, orders, location, edge cases) ----
    FAQ_WORDS_SINGLE = [
        # Delivery
        "delivery", "deliver", "shipping", "shipped", "dispatched", "courier",
        # COD / Payment
        "cod", "cash", "payment", "pay", "paying", "advance", "prepaid",
        # Warranty / Returns
        "warranty", "return", "returns", "replace", "replacement", "exchange", "refund",
        # Store / Location
        "location", "store", "shop", "address", "visit", "located", "branch", "office",
        # Contact
        "contact", "whatsapp", "call", "email", "number",
        # Order process
        "cancel", "cancellation", "track", "tracking",
        # Product condition
        "original", "copy", "fake", "used", "new", "secondhand", "refurbished",
        "boxpack", "sealed", "accessories", "included", "charger", "earphones",
        # Edge cases
        "receive", "someone", "reachable",
        # Urdu keywords
        "ڈیلیوری", "لوکیشن", "وارنٹی", "واپسی", "ادائیگی",
    ]
    FAQ_PHRASES = [
        "cash on delivery", "next day delivery", "next-day delivery",
        "how long", "how many days", "delivery charge", "delivery fee", "delivery time",
        "delivery cost", "do you deliver", "do you offer",
        "pay after", "pay in advance", "need to pay",
        "box pack", "in the box", "come with",
        "physical store", "can i return", "can i cancel",
        "not available", "not reachable", "someone else",
        "what if", "what happens",
        "replacement policy", "return policy", "ordering process",
        "how can i place", "how does the ordering", "what details",
        "do you give", "do i need",
        "outside uae", "all cities",
    ]

    has_order  = any(w in words for w in ORDER_WORDS)
    has_product_cat = any(w in words for w in PRODUCT_CAT_WORDS)
    has_price_single = any(w in words for w in PRICE_WORDS_SINGLE)
    has_price_phrase = any(phrase in user_message_lower for phrase in PRICE_PHRASES)
    has_price = has_price_single or has_price_phrase
    has_faq_single = any(w in words for w in FAQ_WORDS_SINGLE)
    has_faq_phrase = any(phrase in user_message_lower for phrase in FAQ_PHRASES)
    has_faq = has_faq_single or has_faq_phrase

    is_order_intent = has_order
    is_product_query = has_product_cat or (has_price and has_product_cat)

    # CRITICAL FIX: FAQ intent overrides product intent when there is NO actual product brand name
    # e.g. "Is cash on delivery available?" has no product brand → FAQ
    # e.g. "Is Samsung S25 available?" has a product brand → product query
    intent_is_faq = has_faq and not has_product_cat

    # Even if price keyword is present, if it's about delivery/COD charges, it's FAQ
    if has_price and not has_product_cat and has_faq:
        intent_is_faq = True
        is_product_query = False

    in_details_step = sender in USER_STATES and USER_STATES[sender]["step"] == "awaiting_details"

    # is_strong_product_query: only when BOTH a product brand AND an order/price intent are present
    # REMOVED the bare "available" check that was misclassifying FAQ queries
    is_strong_product_query = (has_product_cat and has_order) or (has_price and has_product_cat)

    # Pre-flight Tool Call: Only do product lookup if we actually suspect a product query
    # Skip product lookup entirely for pure FAQ queries to avoid false positives
    if is_product_query or is_strong_product_query:
        p_lookup = product_lookup_tool(user_message_lower)
        product_exists = (p_lookup.get("status") == "found")
    else:
        p_lookup = {"status": "not_found"}
        product_exists = False

    # --- 4. MESSAGE-BY-MESSAGE STATE OVERRIDES (RULE 1 & 3 & 5) ---
    # If user is in ANY active order step, do NOT abort — let the order flow handle it.
    # Words like "number", "location" in the user's details message are NOT FAQ queries.
    # Users can say "cancel" or send a greeting to exit the order flow.

    # --- 5. PRODUCT INTENT EVALUATION (RULE 2 & 4) ---
    # Only trigger this block if we identified a genuine product query (brand name present)
    # SKIP if user is in any active order step — let section 7 handle it to preserve order flow
    in_order_flow = sender in USER_STATES and USER_STATES[sender].get("step") in ("awaiting_details", "awaiting_product", "checking_product", "awaiting_confirmation")
    if is_product_query and not intent_is_faq and not in_order_flow:
        if not product_exists:
            # ANTI-HALLUCINATION GUARD
            if not in_details_step or is_strong_product_query:
                err_msg = "Sorry, this product is not available in our catalog."
                if user_lang == "ur": err_msg = "معذرت، یہ پروڈکٹ ہمارے کیٹلاگ میں دستیاب نہیں ہے۔"
                elif user_lang == "ru": err_msg = "Maafi chahte hain, yeh product hamare catalog mein available nahi hai."
                return err_msg, user_lang
        else:
            # SUCCESSFUL PRODUCT FOUND
            if not in_details_step and not intent_is_faq:
                p_name = p_lookup["name"]
                p_price = p_lookup["price"]
                ans = f"The {p_name} is available for {p_price} AED."
                
                if user_lang == "ur":
                    ans = f"جی ہاں، {p_name} دستیاب ہے۔ اس کی قیمت {p_price} درہم ہے۔"
                elif user_lang == "ru":
                    ans = f"Jee haan, {p_name} available hai. Iski price {p_price} AED hai."
                    
                if is_order_intent:
                    # They asked to "buy X". Transition directly into order collection!
                    USER_STATES[sender] = {"step": "awaiting_details", "product": p_name}
                    ans += "\n\nTo place an order for this, please share your Full Name, Mobile Number, and Google Maps Location."
                    if user_lang == "ur": ans += "\n\nاس کا آرڈر دینے کے لیے براہ کرم اپنا پورا نام، موبائل نمبر، اور گوگل میپس لوکیشن بتائیں۔"
                    elif user_lang == "ru": ans += "\n\nIska order dene ke liye apna poora naam, mobile number, aur location batayein."
                
                return ans, user_lang

    # --- 6. GENERIC ORDER INTENT ("How to order?") ---
    if is_order_intent and sender not in USER_STATES and not intent_is_faq:
        USER_STATES[sender] = {"step": "awaiting_details"}
        if user_lang == "ur":
            return "آرڈر دینے کے لیے براہ کرم اپنا پورا نام، موبائل نمبر، اور گوگل میپس لوکیشن بتائیں۔", user_lang
        elif user_lang == "ru":
            return "Order dene ke liye barah meherbani apna poora naam, mobile number, aur Google Maps location batayein.", user_lang
        return "To place an order, please share your Full Name, Mobile Number, and Google Maps Location.", user_lang

    # --- 7. ACTIVE ORDER FLOW PROCESSING (RULE 6) ---
    # IMPORTANT: If user is in an active order flow, ALWAYS process here
    # regardless of FAQ keywords (e.g. "number", "location" in their details message)
    if sender in USER_STATES:
        state = USER_STATES[sender]
        
        if state["step"] == "awaiting_details":
            has_digit = bool(re.search(r'\d', user_message))
            has_alpha = bool(re.search(r'[a-zA-Z\u0600-\u06FF]{2,}', user_message))
            is_valid_details = has_digit and has_alpha and len(user_message_lower) > 5
            
            if is_valid_details:
                # Translate non-English input to English first
                translated_msg = user_message
                try:
                    if any(ord(c) > 127 for c in user_message):
                        translated_msg = GoogleTranslator(source='auto', target='en').translate(user_message)
                except:
                    pass
                
                # Extract structured details: name, phone, location
                parsed = _extract_customer_details(translated_msg)
                USER_STATES[sender]["details"] = user_message.strip()
                USER_STATES[sender]["customer_name"] = parsed["name"]
                USER_STATES[sender]["customer_phone"] = parsed["phone"]
                USER_STATES[sender]["customer_location"] = parsed["location"]

                if "product" in state and state["product"]:
                    # We already know what they want! Jump to confirmation.
                    USER_STATES[sender]["step"] = "awaiting_confirmation"
                    p_name = state["product"]
                    c_name = parsed["name"]
                    c_phone = parsed["phone"]
                    c_loc = parsed["location"]
                    
                    if user_lang == "ur":
                        return (f"آپ کے آرڈر کی تفصیل یہ ہے:\n"
                                f"پروڈکٹ: {p_name}\n"
                                f"نام: {c_name}\n"
                                f"فون نمبر: {c_phone}\n"
                                f"لوکیشن: {c_loc}\n\n"
                                f"آرڈر کنفرم کرنے کے لیے 'جی ہاں' لکھیں۔"), user_lang
                    elif user_lang == "ru":
                        return (f"Aap ke order ki detail yeh hai:\n"
                                f"Product: {p_name}\n"
                                f"Name: {c_name}\n"
                                f"Phone: {c_phone}\n"
                                f"Location: {c_loc}\n\n"
                                f"Order confirm karne ke liye 'Haan' likhein."), user_lang
                    return (f"Here are your order details:\n"
                            f"Product: {p_name}\n"
                            f"Name: {c_name}\n"
                            f"Phone: {c_phone}\n"
                            f"Location: {c_loc}\n\n"
                            f"Please reply with 'Yes' to confirm your order."), user_lang
                else:
                    USER_STATES[sender]["step"] = "awaiting_product"
                    if user_lang == "ur": return "شکریہ! آپ کیا آرڈر کرنا چاہتے ہیں؟ براہ کرم پروڈکٹ کا نام بتائیں۔", user_lang
                    elif user_lang == "ru": return "Shukriya! Aap kya order karna chahte hain? Barah e meherbani product ka naam batayein.", user_lang
                    return "Thank you. What order would you like to place? (Please specify the product)", user_lang
            else:
                if user_lang == "ur": return "براہ کرم اپنی مکمل تفصیل دیں: پورا نام، موبائل نمبر، اور گوگل میپس لوکیشن۔", user_lang
                elif user_lang == "ru": return "Barah meherbani apni mukammal details dein: Poora Naam, Mobile Number, aur Google Maps Location.", user_lang
                return "Please provide your complete details: Full Name, Mobile Number, and Google Maps Location.", user_lang

        elif state["step"] == "awaiting_product":
            # Clean common intent prefixes so we only check/show the product name
            clean_product = user_message.lower()
            prefixes = [r"\bi want to buy\b", r"\bi want to place an order for\b", r"\bi want to order\b", r"\bi want\b", r"\bi need\b", r"\bbuy\b", r"\border\b", r"\bmujhe\b", r"\bchahiye\b", r"\bmangwana hai\b", r"\bfor\b"]
            for p in prefixes:
                clean_product = re.sub(p, "", clean_product, flags=re.IGNORECASE)
            
            clean_product = clean_product.strip()
            if not clean_product:
                clean_product = user_message.strip()
            
            clean_product = clean_product.title()
            
            # User just told us which product they want — save it and ask them to wait
            USER_STATES[sender]["step"] = "checking_product"
            USER_STATES[sender]["requested_product"] = clean_product
            
            if user_lang == "ur":
                return f"میں '{clean_product}' کی دستیابی چیک کر رہا ہوں۔ براہ کرم 'اوکے' لکھیں تاکہ میں آگے بڑھوں۔", user_lang
            elif user_lang == "ru":
                return f"Main '{clean_product}' ki availability check kar raha hoon. 'OK' likhein taake main aagay barhoon.", user_lang
            return f"Let me check if '{clean_product}' is available. Please reply 'OK' to proceed.", user_lang

        elif state["step"] == "checking_product":
            # User replied (ok/anything) — now actually check the database
            requested = state.get("requested_product", "")
            p_lookup_order = product_lookup_tool(requested.lower())
            if p_lookup_order.get("status") == "found":
                p_name = p_lookup_order["name"]
                p_price = p_lookup_order["price"]
                USER_STATES[sender]["step"] = "awaiting_confirmation"
                USER_STATES[sender]["product"] = p_name
                c_name = state.get('customer_name', 'Customer')
                c_phone = state.get('customer_phone', 'Not provided')
                c_loc = state.get('customer_location', 'Not provided')
                
                if user_lang == "ur":
                    return (f"پروڈکٹ دستیاب ہے! آپ کے آرڈر کی تفصیل یہ ہے:\n"
                            f"پروڈکٹ: {p_name} ({p_price} درہم)\n"
                            f"نام: {c_name}\n"
                            f"فون نمبر: {c_phone}\n"
                            f"لوکیشن: {c_loc}\n\n"
                            f"آرڈر کنفرم کرنے کے لیے 'جی ہاں' لکھیں۔"), user_lang
                elif user_lang == "ru":
                    return (f"Product available hai! Aap ke order ki detail yeh hai:\n"
                            f"Product: {p_name} ({p_price} AED)\n"
                            f"Name: {c_name}\n"
                            f"Phone: {c_phone}\n"
                            f"Location: {c_loc}\n\n"
                            f"Order confirm karne ke liye 'Haan' likhein."), user_lang
                return (f"Product is available! Here are your order details:\n"
                        f"Product: {p_name} ({p_price} AED)\n"
                        f"Name: {c_name}\n"
                        f"Phone: {c_phone}\n"
                        f"Location: {c_loc}\n\n"
                        f"Please reply with 'Yes' to confirm your order."), user_lang
            else:
                err_msg = f"Sorry, '{requested}' is not available in our catalog."
                if user_lang == "ur": err_msg = f"معذرت، '{requested}' ہمارے کیٹلاگ میں دستیاب نہیں ہے۔"
                elif user_lang == "ru": err_msg = f"Maafi chahte hain, '{requested}' hamare catalog mein available nahi hai."
                del USER_STATES[sender]
                return err_msg, user_lang

        elif state["step"] == "awaiting_confirmation":
            confirm_words = ["yes", "y", "confirm", "haan", "ha", "ji", "theek", "ok", "جی ہاں", "ہاں", "جی", "ٹھیک ہے", "تصدیق"]
            cancel_words = ["no", "cancel", "nahi", "nhi", "na", "نہیں", "نہی", "منسوخ"]
            
            if any(w in user_message_lower for w in confirm_words):
                CONFIRMED_ORDERS_QUEUE[sender] = {
                    "product": state.get("product", "Unknown Product"),
                    "details": state.get("details", ""),
                    "customer_name": state.get("customer_name", "Customer"),
                    "customer_phone": state.get("customer_phone", ""),
                    "customer_location": state.get("customer_location", "")
                }
                del USER_STATES[sender]
                if user_lang == "ur": return "آپ کا آرڈر کنفرم ہو گیا ہے! کامیابی سے جمع کر دیا گیا۔ آپ کو 3-5 کاروباری دنوں میں ڈیلیوری ہو جائے گی۔", user_lang
                elif user_lang == "ru": return "Aap ka order confirm ho gaya! 3-5 kaam ke dinon mein delivery ho jayegi.", user_lang
                return "Order Confirmed! Your order has been placed successfully. You can expect delivery within 3-5 working days.", user_lang
            elif any(w in user_message_lower for w in cancel_words):
                del USER_STATES[sender]
                if user_lang == "ur": return "آرڈر منسوخ کر دیا گیا۔", user_lang
                elif user_lang == "ru": return "Order cancel kar diya gaya.", user_lang
                return "Order cancelled. How else can I help you?", user_lang
            else:
                if user_lang == "ur": return "آرڈر تصدیق کرنے کے لیے 'جی ہاں' لکھیں۔", user_lang
                elif user_lang == "ru": return "Confirm karne ke liye 'Haan' likhein.", user_lang
                return "Please reply with 'Yes' to confirm your order, or 'No' to cancel.", user_lang

    # --- 8. FAQ FALLBACK VIA FAQ TOOL ---
    # Pass the product lookup result to the FAQ tool so it knows if a product search was already attempted
    tool_results = f"Product Search Status: {p_lookup.get('status')}"
    if product_exists:
        tool_results += f"\nProduct Found: {p_lookup.get('name')} at {p_lookup.get('price')} AED"

    return faq_lookup_tool(user_message, user_lang, tool_results=tool_results, history=session_history, long_term_context=long_term_context), user_lang
