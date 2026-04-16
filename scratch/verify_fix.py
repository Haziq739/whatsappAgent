import re

def clean_details_test(user_message):
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
    return clean_details

# Test cases
test_cases = [
    "Mera naam Shah, mobile number 0928792822 aur location Dubai",
    "Name Haziq, Number 123456, Location Riyadh",
    "Mera naam Ali, number 987654 aur meri location Abu Dhabi",
    "location Dubai"
]

for tc in test_cases:
    print(f"Input: {tc}")
    print(f"Output: {clean_details_test(tc)}")
    print("-" * 20)
