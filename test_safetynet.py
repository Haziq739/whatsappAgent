from utils.brain import retrieve_context, tokenizer, model, generator, SYSTEM_PROMPT
import re

def test_safety_net(user_message, user_lang="en"):
    context = retrieve_context(user_message, top_k=5)
    print("--- CONTEXT ---")
    print(context)
    print("---------------")
    
    answer = "" # Let's simulate a failed LLM response
    is_dont_know = True
    
    price_keywords_en = ["price", "cost", "how much"]
    price_keywords_ur = ["kitne", "keemat", "price", "rate", "کتنے", "قیمت", "ریٹ"]
    is_price_query = (
        any(kw in user_message.lower() for kw in price_keywords_en) or
        any(kw in user_message.lower() for kw in price_keywords_ur)
    )

    list_keywords_en = ["products", "models", "stock", "have", "list", "phones"]
    list_keywords_ur = ["products", "کون", "کونسے", "کیا", "ہے", "پاس", "available"]
    is_catalog_query = (
        any(kw in user_message.lower() for kw in list_keywords_en) or
        any(kw in user_message.lower() for kw in list_keywords_ur)
    )

    print(f"is_price_query: {is_price_query}")
    print(f"is_catalog_query: {is_catalog_query}")

    if (len(answer) < 5 or is_dont_know):
        if is_catalog_query:
            catalog_line = ""
            for line in context.split("\n"):
                if "available" in line.lower() or "products" in line.lower() or "موبائلز" in line:
                    catalog_line = line
                    break
            if catalog_line:
                print("CATALOG FALLBACK TRIGGERED!")
                return catalog_line, user_lang

        if is_price_query:
            clean_msg = re.sub(r'[^\w\s]', '', user_message.lower())
            query_terms = [t for t in clean_msg.split() if len(t) > 2]
            print(f"query_terms: {query_terms}")
            
            context_price = ""
            best_line = ""
            max_matches = 0
            for line in context.split("\n"):
                if not line.strip(): continue
                matches = sum(1 for term in query_terms if term in line.lower())
                if matches > max_matches:
                    max_matches = matches
                    best_line = line
            
            print(f"best_line: {best_line}")
            
            if best_line:
                price_patterns = [
                    r"(\d+\s*AED)", r"(price\s+is\s+\d+)", r"(\d+\s*درہم)"
                ]
                for pattern in price_patterns:
                    matches = re.findall(pattern, best_line, re.IGNORECASE)
                    if matches:
                        context_price = matches[0]
                        break
            
            print(f"context_price: {context_price}")
            
            if context_price:
                return f"The price for this product is {context_price}.", user_lang
    
    return "FALLBACK", user_lang

test_safety_net("Redmi 14c price?")
print("\n" + "="*40 + "\n")
test_safety_net("S25 Ultra kitne ka hai?", "ru")
print("\n" + "="*40 + "\n")
test_safety_net("What products do you have?")
