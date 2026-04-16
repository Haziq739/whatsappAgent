from utils.brain import generate_rag_response, USER_STATES
import re

# Clear any leftover states
USER_STATES.clear()

queries = [
    # --- The user's specific failing cases ---
    ("Order dene ka tareeqa kya hai?", "ru", "RU order intent with kya hai"),
    ("Mughy samsung galaxy s25 ultra chhaye", "ru", "RU extraction cleanliness 1"),
    ("Mujhy iPhone 15 mangwana hai h", "ru", "RU extraction cleanliness 2"),
    ("I want HP Laptop", "en", "EN order intent"),
]

print("--- SURGICAL RE-VERIFICATION (ROUND 2) ---\n")
for q, expected_lang, label in queries:
    USER_STATES.clear()  # Reset between tests
    response, lang = generate_rag_response(q, sender="test_user")
    lang_ok = "OK" if lang == expected_lang else f"MISMATCH (Got {lang})"
    
    # Check if RU response is actually RU
    is_order_intent = "Order dene ke liye" in response or "share your Full Name" in response or "apna poora naam" in response
    resp_lang_ok = "PASS" if (expected_lang == "ru" and ("Order dene ke liye" in response or "apna poora naam" in response)) or (expected_lang == "en" and "share your Full Name" in response) else "FAIL"

    print(f"[{label}] QUERY: {q}")
    print(f"  DETECTED LANG: {lang} ({lang_ok})")
    print(f"  IS ORDER INTENT: {is_order_intent} ({resp_lang_ok})")
    print(f"  RESPONSE: {response}")
    print("-" * 60)

# --- Test product name cleaning specifically for the "chhaye" case ---
print("\n--- SPECIFIC PRODUCT EXTRACTION TEST ---")
cases = [
    ("Mughy samsung galaxy s25 ultra chhaye", "Samsung Galaxy S25 Ultra"),
    ("Samsung S24 Ultra book karna hai h", "Samsung S24 Ultra"),
    ("I want to buy the HP Laptop", "Hp Laptop"),
]

for msg, expected in cases:
    USER_STATES.clear()
    USER_STATES["test_user"] = {"step": "awaiting_product", "details": "Danish, 123456, Dubai"}
    response, lang = generate_rag_response(msg, sender="test_user")
    
    product_line = ""
    for line in response.split("\n"):
        if "Product:" in line:
            product_line = line.split("Product:")[1].strip()

    extraction_ok = "PASS" if product_line == expected else f"FAIL (Got '{product_line}' expected '{expected}')"
    print(f"  INPUT: '{msg}'")
    print(f"  EXTRACTED PRODUCT: '{product_line}'")
    print(f"  RESULT: {extraction_ok}")
    print("-" * 60)
