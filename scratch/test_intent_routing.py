"""
Test script to verify intent routing in brain.py.
This tests ONLY the intent classification logic without loading the LLM model.
We mock the faq_lookup_tool and product_lookup_tool to isolate the routing.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---- Mock the heavy imports BEFORE importing brain ----
import types

# Mock product_tool
mock_product_tool = types.ModuleType("tools.product_tool")
def mock_product_lookup(query):
    """Always returns not_found so we can verify FAQ routing doesn't hit product path."""
    return {"status": "not_found"}
mock_product_tool.product_lookup_tool = mock_product_lookup
sys.modules["tools.product_tool"] = mock_product_tool

# Mock faq_tool
mock_faq_tool = types.ModuleType("tools.faq_tool")
def mock_faq_lookup(user_message, user_lang="en", tool_results=None, history=None):
    return f"[FAQ_TOOL_CALLED] {user_message}"
mock_faq_tool.faq_lookup_tool = mock_faq_lookup
sys.modules["tools.faq_tool"] = mock_faq_tool

# Mock order_tool
mock_order_tool = types.ModuleType("tools.order_tool")
def mock_order_creation(customer_number, raw_details, product_name):
    return True
mock_order_tool.order_creation_tool = mock_order_creation
sys.modules["tools.order_tool"] = mock_order_tool

# Mock firebase_db
mock_firebase_db = types.ModuleType("utils.firebase_db")
def mock_get_recent_history(sender, limit=5):
    return ""
mock_firebase_db.get_recent_history = mock_get_recent_history
sys.modules["utils.firebase_db"] = mock_firebase_db

# Mock langdetect
mock_langdetect = types.ModuleType("langdetect")
def mock_detect(text):
    return "en"
mock_langdetect.detect = mock_detect
sys.modules["langdetect"] = mock_langdetect

# Mock deep_translator
mock_translator_mod = types.ModuleType("deep_translator")
class MockGoogleTranslator:
    def __init__(self, source='auto', target='en'):
        pass
    def translate(self, text):
        return text
mock_translator_mod.GoogleTranslator = MockGoogleTranslator
sys.modules["deep_translator"] = mock_translator_mod

# NOW import brain
from utils.brain import generate_rag_response

# ---- TEST CASES ----
FAQ_QUESTIONS = [
    # Delivery FAQs
    ("How long does delivery take?", True),
    ("Do you deliver in Dubai?", True),
    ("What is the delivery time for orders?", True),
    ("How much is delivery charge?", True),
    ("Do you offer next-day delivery?", True),
    # Payment / COD FAQs
    ("Is cash on delivery available?", True),
    ("Can I pay after checking the phone?", True),
    ("Do I need to pay in advance?", True),
    ("Is COD available in all cities?", True),
    # Product Condition FAQs
    ("Are the phones new or used?", True),
    ("Is this original or copy?", True),
    ("Do the phones come in box pack?", True),
    ("What accessories are included in the box?", True),
    # Return / Warranty FAQs
    ("Do you give warranty?", True),
    ("What if the phone has an issue?", True),
    ("Can I return the product?", True),
    ("What is your replacement policy?", True),
    ("What happens if I don't like the phone?", True),
    # Order Process FAQs
    ("How can I place an order?", True),
    ("What details do you need for order?", True),
    ("How does the ordering process work?", True),
    ("Can I cancel my order?", True),
    # Location / Contact FAQs
    ("Where are you located?", True),
    ("Do you have a physical store?", True),
    ("What is your WhatsApp number?", True),
    ("How can I contact you?", True),
    # Edge Case FAQs
    ("What if I am not available at delivery time?", True),
    ("Can someone else receive my order?", True),
    ("What if my number is not reachable?", True),
    ("Do you deliver outside UAE?", True),
]

DOMAIN_GUARDRAIL_QUESTIONS = [
    ("When is the quaid-e-azam birthday", False),
    ("Who is the president of Pakistan?", False),
    ("What is the capital of France?", False),
]

print("=" * 80)
print("INTENT ROUTING TEST — FAQ Questions")
print("=" * 80)

passed = 0
failed = 0

for question, should_be_faq in FAQ_QUESTIONS:
    response, lang = generate_rag_response(question, sender="test_user_faq")
    is_faq = response.startswith("[FAQ_TOOL_CALLED]")
    is_product_err = "not available in our catalog" in response
    
    # For FAQ questions, we want them to go to the FAQ tool, NOT get "not available in catalog"
    if should_be_faq:
        if is_faq:
            status = "[PASS] PASS"
            passed += 1
        elif is_product_err:
            status = "[FAIL] FAIL (Misrouted to Product -> 'not available in catalog')"
            failed += 1
        else:
            # Could be a greeting or order response, check
            status = f"[WARN] UNEXPECTED: {response[:80]}"
            failed += 1
    
    print(f"  {status} | Q: \"{question}\"")
    if not is_faq and should_be_faq:
        print(f"           -> Got: \"{response[:100]}\"")

print(f"\n{'=' * 80}")
print(f"DOMAIN GUARDRAIL TEST — Off-Topic Questions")
print(f"{'=' * 80}")

for question, should_be_faq in DOMAIN_GUARDRAIL_QUESTIONS:
    response, lang = generate_rag_response(question, sender="test_user_domain")
    is_faq = response.startswith("[FAQ_TOOL_CALLED]")
    
    # These should go to FAQ tool (which has its own domain guardrail), that's fine
    # The key thing is they should NOT get routed to product error
    is_product_err = "not available in our catalog" in response
    if not is_product_err:
        print(f"  [PASS] PASS | Q: \"{question}\" -> Routed to FAQ (domain guard will handle)")
        passed += 1
    else:
        print(f"  [FAIL] FAIL | Q: \"{question}\" -> Misrouted to product error")
        failed += 1

print(f"\n{'=' * 80}")
print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed} tests")
print(f"{'=' * 80}")
