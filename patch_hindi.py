"""
Applies Devanagari Hindi mappings to brain.py to handle Whisper's Hindi output for Urdu speech.
"""
import re

path = 'utils/brain.py'
content = open(path, 'r', encoding='utf-8').read()

# 1. Early Language Detection
old1 = "    URDU_SCRIPT_CHARS = re.compile(r'[\\u0600-\\u06FF]')"
new1 = "    # Match Urdu (Arabic) or Hindi (Devanagari) script\n    URDU_SCRIPT_CHARS = re.compile(r'[\\u0600-\\u06FF\\u0900-\\u097F]')"
if old1 in content: content = content.replace(old1, new1)

# 2. Greetings
old2 = "        \"سلام\", \"السلام\", \"علیکم\", \"وعلیکم\", \"السلامعلیکم\", \"وعلیکماسلام\""
new2 = "        \"سلام\", \"السلام\", \"علیکم\", \"وعلیکم\", \"السلامعلیکم\", \"وعلیکماسلام\",\n        \"नमस्ते\", \"नमस्कार\", \"हेलो\", \"सलाम\" # Devanagari Hindi"
if old2 in content: content = content.replace(old2, new2)

# 3. Order Words
old3 = "        \"\u0622\u0631\u0688\u0631\", \"\u062e\u0631\u06cc\u062f\", \"\u0645\u0646\u06af\u0648\u0627\","
new3 = "        \"\u0622\u0631\u0688\u0631\", \"\u062e\u0631\u06cc\u062f\", \"\u0645\u0646\u06af\u0648\u0627\",\n        \"ऑर्डर\", \"बुक\", \"मंगवा\", \"खरीद\","
if old3 in content: content = content.replace(old3, new3)

# 4. Intent Words
old4 = "        \"ha\", \"hain\", \"\u06a9\u06cc\u0633\u06d2\", \"\u06a9\u0631\u0646\u0627\", \"\u0637\u0631\u06cc\u0642\u06c1\", \"\u06a9\u0631\u06cc\u06ba\", \"\u062f\u06cc\u0646\u0627\","
new4 = "        \"ha\", \"hain\", \"\u06a9\u06cc\u0633\u06d2\", \"\u06a9\u0631\u0646\u0627\", \"\u0637\u0631\u06cc\u0642\u06c1\", \"\u06a9\u0631\u06cc\u06ba\", \"\u062f\u06cc\u0646\u0627\",\n        \"कैसे\", \"करना\", \"तरीका\", \"चाहिए\", \"क्या\", \"हैं\", \"है\","
if old4 in content: content = content.replace(old4, new4)

# 5. Confirmation Words
old5 = '            confirm_words = ["yes", "y", "confirm", "haan", "جی ہاں", "ہاں", "جی", "ٹھیک ہے", "تصدیق"]\n            cancel_words = ["no", "cancel", "nahi", "نہیں", "نہی", "منسوخ"]'
new5 = '            confirm_words = ["yes", "y", "confirm", "haan", "جی ہاں", "ہاں", "جی", "ٹھیک ہے", "تصدیق", "हाँ", "जी"]\n            cancel_words = ["no", "cancel", "nahi", "نہیں", "نہی", "منسوخ", "नहीं", "कैंसिल"]'
if old5 in content: content = content.replace(old5, new5)

# 6. Price Queries
old6 = '    price_keywords_ur = ["kitne", "keemat", "price", "rate", "کتنے", "قیمت", "ریٹ"]'
new6 = '    price_keywords_ur = ["kitne", "keemat", "price", "rate", "کتنے", "قیمت", "ریٹ", "कितने", "कीमत", "रेट"]'
if old6 in content: content = content.replace(old6, new6)

# 7. Don't Know
old7 = '    dont_know_variants = ["i don\'t know", "mujhe nahi maloom", "i do not know", "i am not sure", "مجھے نہیں معلوم", "معلوم نہیں"]'
new7 = '    dont_know_variants = ["i don\'t know", "mujhe nahi maloom", "i do not know", "i am not sure", "مجھے نہیں معلوم", "معلوم نہیں", "मुझे नहीं पता", "मुझे नहीं मालूम"]'
if old7 in content: content = content.replace(old7, new7)

open(path, 'w', encoding='utf-8').write(content)
print('Hindi mapping injected into brain.py OK')
