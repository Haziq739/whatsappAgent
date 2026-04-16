import re

def test_extract_product(user_message):
    user_message_lower = user_message.lower().strip()
    phrases_to_remove = [
        r"(?i)\bi\s+want\s+to\s+buy\b\s*", r"(?i)\bi\s+want\s+to\s+order\b\s*", r"(?i)\bi\s+would\s+like\s+to\s+order\b\s*",
        r"(?i)\bi\s+would\s+like\b\s*", r"(?i)\bi\s+want\b\s*", r"(?i)\bi\s+need\b\s*", r"(?i)\bgive\s+me\b\s*",
        r"(?i)\bmujh[yeai]+\b\s*", r"(?i)\bmugh[yeai]+\b\s*",
        r"(?i)\boff?er\b\s*", r"(?i)\bpurchase\b\s*", r"(?i)\bbuy\b\s*", r"(?i)\bget\b\s*", r"(?i)\bto\b\s*",
        r"(?i)\border\b\s*", r"(?i)\ban?\b\s*", r"(?i)\bthe\b\s*", r"(?i)\bor\b\s*",
        r"(?i)\s+cha[hyei]+[yei]*\b",
        r"(?i)\s+dedo\b", r"(?i)\s+order\s+karna\s+hai\b", r"(?i)\s+lena\s+hai\b", 
        r"(?i)\s+mangwana\s+hai\b", r"(?i)\bplease\b\s*", r"(?i)\s+please\b", r"(?i)\bi\s+am\s+looking\s+for\b\s*"
    ]
    clean_product = user_message_lower
    for pattern in phrases_to_remove:
        clean_product = re.sub(pattern, "", clean_product)
    
    return clean_product.strip().title()

test_msg = "Mughy Samsung mobile chahye"
result = test_extract_product(test_msg)
print(f"Input: {test_msg}")
print(f"Extracted: {result}")

test_msg_2 = "mujhe Redmi phone chahiye"
result_2 = test_extract_product(test_msg_2)
print(f"Input: {test_msg_2}")
print(f"Extracted: {result_2}")
