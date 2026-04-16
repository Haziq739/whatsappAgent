import sys
import os
sys.path.append(os.getcwd())

from utils.brain import generate_rag_response, USER_STATES

test_phrases = [
    "order place karny ka kya tareqa ha",
    "order karne ka kia tareqa",
    "Order kaise karein",
    "order karne ka tareqqa kya ha",
    "order place karne ka tareeka kia ha",
    "bhai muje order karna hai"
]

print("--- Testing Roman Urdu Intent Detection Trigger ---")
for phrase in test_phrases:
    # Reset state for each test to check the initial trigger
    USER_STATES.clear()
    
    response, lang = generate_rag_response(phrase, sender="test_user")
    print(f"Phrase: '{phrase}'")
    print(f"Detected Lang: {lang}")
    print(f"Response: {response}")
    print("-" * 30)
