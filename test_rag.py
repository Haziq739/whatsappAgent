from utils.brain import generate_rag_response
import os

# Create/Clear debug log
if os.path.exists("debug_log.txt"):
    os.remove("debug_log.txt")

print("Testing generate_rag_response directly...")
query = "S25 Ultra price?"
sender = "test_user"

response, lang = generate_rag_response(query, sender=sender)

print(f"\nQUERY: {query}")
print(f"LANG: {lang}")
print(f"RESPONSE: {response}")

if os.path.exists("debug_log.txt"):
    print("\n--- DEBUG LOG CONTENT ---")
    with open("debug_log.txt", "r", encoding="utf-8") as f:
        print(f.read())
else:
    print("\n❌ debug_log.txt was not created!")
