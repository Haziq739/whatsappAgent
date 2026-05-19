import re
from utils.firebase_db import db

def product_lookup_tool(query):
    """
    Checks if a product exists in Firebase based on query.
    Returns -> dict(status="found", name, price) OR dict(status="not_found")
    """
    print(f"[TOOL CALLED] ProductLookupTool -> Analyzing query: '{query}'")
    
    if not db:
        print("[TOOL RESULT] Firebase not initialized.")
        return {"status": "not_found", "reason": "db_not_initialized"}
        
    try:
        products_ref = db.collection('products')
        docs = products_ref.get()
        query_cln = query.lower().strip()
        
        # Strip out noisy intent words from the query before matching
        noise_words = [
            "buy", "need", "want", "purchase", "looking", "for", "the", "a", "an", "is", "there", 
            "price", "of", "cost", "how", "much", "available", "stock", "nokia", "samsung", "apple", "iphone",
            "chahiye", "lena", "mangwana", "hai", "kya", "batao", "order", "karna",
            "i", "to", "do", "you", "have", "can", "get", "me", "my", "it", "this", "that"
        ]
        
        # Remove punctuation
        query_words = re.sub(r'[^\w\s]', '', query_cln).split()
        
        # Extract potential nouns (Filter out intent words, but keep brand names if exact)
        filtered_words = [w for w in query_words if w not in [n for n in noise_words if n not in ["samsung", "iphone", "nokia", "apple"]]]
        
        if not filtered_words:
            # If no nouns left, fallback to full string check
            filtered_words = query_words

        # Identify number sequences in query (e.g. ['25', '16'])
        query_digits = set(re.findall(r'\d+', query_cln))

        # Known brands
        major_brands = {"samsung", "apple", "iphone", "google", "pixel", "xiaomi", "redmi", "poco", "macbook", "hp", "dell", "lenovo", "asus"}
        query_brands = {w for w in filtered_words if w in major_brands}

        best_match = None
        highest_score = 0
        
        # Fetch all products into memory for matching (since we have 68 products, this is fast enough)
        for doc in docs:
            p_data = doc.to_dict()
            p_name = p_data.get('name', '').lower()
            p_words = p_name.split()
            
            # STRICT FILTER 1: Brand Mismatch
            p_brands = {w for w in p_words if w in major_brands}
            if query_brands and p_brands:
                if not query_brands.intersection(p_brands):
                    continue # E.g., User asked for Samsung, product is iPhone -> SKIP entirely
            
            p_digits = set(re.findall(r'\d+', p_name))
            
            # STRICT FILTER 2: Model Number Mismatch
            if query_digits and p_digits:
                if not query_digits.intersection(p_digits):
                    continue # Skip products with completely different model numbers
            
            score = 0
            # Award points for specific keyword matches
            for w in filtered_words:
                if len(w) >= 2 and w in p_words:
                    score += 2
                elif len(w) >= 3 and w in p_name:
                    score += 1
            
            # Exact number/model match gets heavy boost
            for w in filtered_words:
                if any(c.isdigit() for c in w):
                    if w in p_words:
                        score += 5
                    elif any(w in pw or pw in w for pw in p_words):
                        score += 3
                        
            if score > highest_score and score >= 2: # Requires at least score 2
                highest_score = score
                best_match = p_data
                
        if best_match:
            print(f"[TOOL RESULT] Product found: {best_match.get('name')}")
            return {
                "status": "found", 
                "name": best_match.get("name"), 
                "price": best_match.get("price"), 
                "stock_status": best_match.get("stock_status")
            }
            
        print(f"[TOOL RESULT] Product NOT found. Missing from catalog.")
        return {"status": "not_found"}
        
    except Exception as e:
        print(f"[TOOL ERROR] Error looking up product: {e}")
        return {"status": "not_found"}

