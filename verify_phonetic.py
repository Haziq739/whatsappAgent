from utils.brain import generate_rag_response, USER_STATES

# Clear any leftover states
USER_STATES.clear()

queries = [
    ("Order dene ka tareeqa kya hai?", "ru", "Standard"),
    ("auder denay ka tareeka kia hai", "ru", "Phonetic 1"),
    ("آرڈر کرنے کا طریقہ کیا ہے؟", "ur", "Urdu Script"),
]

print("--- PHONETIC INTENT VERIFICATION (SAFE PRINT) ---\n")
for q, expected_lang, label in queries:
    USER_STATES.clear()
    response, lang = generate_rag_response(q, sender="test_user")
    is_order_intent = "Order dene ke liye" in response or "apna poora naam" in response or "آرڈر دینے کے لیے" in response
    print(f"[{label}] Query OK") # Avoid raw urdu in console
    print(f"  DETECTED LANG: {lang}")
    print(f"  IS ORDER INTENT: {is_order_intent}")
    print("-" * 50)
