"""
Patch: replace the narrow intent-detection block in brain.py with a broad one,
using line numbers so line-ending format doesn't matter.
"""
import sys

path = "utils/brain.py"
lines = open(path, "r", encoding="utf-8").readlines()

# Find the block start by looking for the comment
start = None
for i, line in enumerate(lines):
    if "# --- INTENT DETECTION" in line:
        start = i
        break

if start is None:
    sys.stdout.buffer.write(b"ERROR: INTENT DETECTION comment not found\n")
    sys.exit(1)

# Find end of the block: first line that starts with '    # 2.' after start
end = None
for i in range(start + 1, len(lines)):
    if "# 2. Get Context" in lines[i]:
        end = i
        break

if end is None:
    sys.stdout.buffer.write(b"ERROR: end of intent block not found\n")
    sys.exit(1)

sys.stdout.buffer.write(f"Replacing lines {start+1} to {end} (0-indexed {start} to {end-1})\n".encode())

# The replacement block (use \n to match whatever line endings the file uses)
new_block = [
    "    # --- INTENT DETECTION (If not in an active order flow) ---\n",
    "    # Keyword-pair: works for Roman Urdu, Arabic-script Urdu, and English.\n",
    "    ORDER_WORDS = [\n",
    "        \"order\", \"khareedna\", \"kharidna\", \"mangwana\", \"book\", \"purchase\", \"buy\",\n",
    "        \"\u0622\u0631\u0688\u0631\", \"\u062e\u0631\u06cc\u062f\", \"\u0645\u0646\u06af\u0648\u0627\",\n",
    "    ]\n",
    "    INTENT_WORDS = [\n",
    "        \"kaise\", \"karna\", \"karo\", \"tareeqa\", \"process\", \"place\", \"chahiye\",\n",
    "        \"dena\", \"how\", \"want\", \"would\", \"kia\", \"kya\", \"procedure\", \"steps\",\n",
    "        \"ha\", \"hain\", \"\u06a9\u06cc\u0633\u06d2\", \"\u06a9\u0631\u0646\u0627\", \"\u0637\u0631\u06cc\u0642\u06c1\", \"\u06a9\u0631\u06cc\u06ba\", \"\u062f\u06cc\u0646\u0627\",\n",
    "    ]\n",
    "    EXACT_ORDER_PHRASES = [\n",
    "        \"process of placing an order\", \"how to place an order\", \"how to order\",\n",
    "        \"place an order\", \"process of order\", \"order process\", \"want to order\",\n",
    "        \"\u0622\u0631\u0688\u0631 \u06a9\u06cc\u0633\u06d2 \u06a9\u0631\u06cc\u06ba\", \"\u0622\u0631\u0688\u0631 \u06a9\u0631\u0646\u0627 \u06c1\u06d2\",\n",
    "        \"\u0622\u0631\u0688\u0631 \u062f\u06cc\u0646\u06d2 \u06a9\u0627 \u0637\u0631\u06cc\u0642\u06c1\", \"\u0622\u0631\u0688\u0631 \u067e\u0644\u06cc\u0633\",\n",
    "    ]\n",
    "\n",
    "    has_order  = any(w in user_message_lower for w in ORDER_WORDS)\n",
    "    has_intent = any(w in user_message_lower for w in INTENT_WORDS)\n",
    "    has_exact  = any(ph in user_message_lower for ph in EXACT_ORDER_PHRASES)\n",
    "\n",
    "    if has_exact or (has_order and has_intent):\n",
    "        USER_STATES[sender] = {\"step\": \"awaiting_details\"}\n",
    "        if user_lang == \"ur\":\n",
    "            return \"\u0622\u0631\u0688\u0631 \u062f\u06cc\u0646\u06d2 \u06a9\u06d2 \u0644\u06cc\u06d2 \u0628\u0631\u0627\u06c1 \u06a9\u0631\u0645 \u0627\u067e\u0646\u0627 \u067e\u0648\u0631\u0627 \u0646\u0627\u0645\u060c \u0645\u0648\u0628\u0627\u0626\u0644 \u0646\u0645\u0628\u0631\u060c \u0627\u0648\u0631 \u06af\u0648\u06af\u0644 \u0645\u06cc\u067e\u0633 \u0644\u0648\u06a9\u06cc\u0634\u0646 \u0628\u062a\u0627\u0626\u06cc\u06ba\u06d4\", user_lang\n",
    "        return \"To place an order, please share your Full Name, Mobile Number, and Google Maps Location.\", user_lang\n",
    "\n",
]

# Stitch: lines before + new block + lines from 'end' onwards
new_lines = lines[:start] + new_block + lines[end:]
open(path, "w", encoding="utf-8").writelines(new_lines)
sys.stdout.buffer.write(b"brain.py patched and saved OK\n")
