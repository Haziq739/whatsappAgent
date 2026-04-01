import chromadb
from sentence_transformers import SentenceTransformer

# 1. Initialize the embedding model (Same one used in brain.py)
print("✨ Initializing embedding engine...")
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# 2. Connect to the database folder
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="business_Knowledge")

# 3. Your Business Data
documents = [
    # --- English ---
    # --- English User Queries (High Priority) ---
    {"id": "en_q_price_1", "text": "What is the price of this phone?"},
    {"id": "en_q_price_2", "text": "Price please"},
    {"id": "en_q_price_3", "text": "How much is S25 Ultra?"},
    {"id": "en_q_price_4", "text": "Latest price of Redmi 14C?"},

    {"id": "en_q_delivery_1", "text": "How long does delivery take?"},
    {"id": "en_q_delivery_2", "text": "Can I get it tomorrow?"},
    {"id": "en_q_delivery_3", "text": "Do you deliver to Dubai?"},
    {"id": "en_q_delivery_4", "text": "Delivery charges?"},

    {"id": "en_q_cod_1", "text": "Is cash on delivery available?"},
    {"id": "en_q_cod_2", "text": "Can I pay after checking the phone?"},
    {"id": "en_q_cod_3", "text": "Do I have to pay first?"},

    {"id": "en_q_original_1", "text": "Is this original or copy?"},
    {"id": "en_q_original_2", "text": "Is it PTA approved?"},
    {"id": "en_q_original_3", "text": "Is the phone brand new?"},

    {"id": "en_q_order_1", "text": "How to place an order?"},
    {"id": "en_q_order_2", "text": "What details do you need for order?"},
    {"id": "en_q_order_3", "text": "How can I order?"},

    {"id": "en_q_warranty_1", "text": "Do you give warranty?"},
    {"id": "en_q_warranty_2", "text": "What if the phone has an issue?"},
    {"id": "en_q_warranty_3", "text": "Replacement policy?"},

    {"id": "en_q_stock_1", "text": "Is this available right now?"},
    {"id": "en_q_stock_2", "text": "Available in stock?"},
    {"id": "en_q_stock_3", "text": "Which colors are available?"},

    {"id": "en_q_location_1", "text": "Where are you located?"},
    {"id": "en_q_location_2", "text": "Can I visit your shop?"},
    {"id": "en_q_location_3", "text": "Do you have a physical store?"},

    {"id": "en_q_contact_1", "text": "Your WhatsApp number?"},
    {"id": "en_q_contact_2", "text": "How can I contact you?"},

    {"id": "en_q_offer_1", "text": "Is the offer still available?"},
    {"id": "en_q_offer_2", "text": "When does this deal expire?"},

    # --- Edge Cases ---
    {"id": "en_edge_1", "text": "What if I am not available at delivery time?"},
    {"id": "en_edge_2", "text": "What if I don't like the phone after checking?"},
    {"id": "en_edge_3", "text": "Can I return the phone?"},
    {"id": "en_edge_4", "text": "Do you deliver outside UAE?"},
    {"id": "en_edge_5", "text": "What if my number is not reachable?"},
    {"id": "en_edge_6", "text": "Can someone else receive my order?"},
    {"id": "en_edge_7", "text": "Do you accept online payment?"},
    {"id": "en_edge_8", "text": "What if the product is damaged?"},
    {"id": "en_edge_9", "text": "Is there any hidden charges?"},
    {"id": "en_edge_10", "text": "Can I cancel my order?"},

    {"id": "en_1", "text": "We deliver all orders within 2 to 3 working days."},
    {"id": "en_2", "text": "You can track your order using your phone number or order ID."},
    {"id": "en_3", "text": "Cash on Delivery is available across the UAE."},
    {"id": "en_4", "text": "Our phones come with a 7-day replacement warranty."},
    {"id": "en_5", "text": "For more information, contact us on WhatsApp or call 0563235075."},
    {"id": "en_redmi14c_price", "text": "Redmi 14c price is 210 AED."},
    {"id": "en_price_s25_ultra", "text": "S25 ULTRA is priced at 210 AED as part of a 24-hour flash offer."},
    {"id": "en_price_redmi_14c_5g", "text": "Redmi 14c 5G price is 159 AED or 210 AED depending on the specific promotion variant."},
    {"id": "en_price_s24_ultra_copy", "text": "Samsung S24 Ultra (High Quality Copy) is available for 375 AED."},
    {"id": "en_price_s23_ultra", "text": "S23 Ultra is available for 210 AED."},
    {"id": "en_price_generic_ultra", "text": "Ultra series models are generally priced around 210 AED during promotional periods."},
    {"id": "en_spec_fingerprint", "text": "The phones feature a fingerprint sensor located inside/on the screen."},
    {"id": "en_spec_condition", "text": "All phones are brand new, seal-packed, and come in the original box pack."},
    {"id": "en_spec_originality", "text": "The gadgets are described as original quality products."},
    {"id": "en_spec_contents", "text": "The phone box includes a charger and all standard accessories."},
    {"id": "en_delivery_area", "text": "We provide home delivery across the UAE, including Dubai, Ajman, and Jebel Ali."},
    {"id": "en_delivery_time_standard", "text": "Standard delivery takes 2 to 3 working days."},
    {"id": "en_delivery_time_fast", "text": "Next-day delivery is available for orders confirmed before the cutoff time (Tomorrow you received)."},
    {"id": "en_delivery_charges", "text": "Delivery charges are typically 10 AED, though free delivery is offered on specific deals."},
    {"id": "en_delivery_process", "text": "The rider will contact you via phone call before arriving at your location."},
    {"id": "en_loc_ajman", "text": "Ags Gadgets has a presence/base in Ajman."},
    {"id": "en_loc_jebel_ali", "text": "Our warehouse is located in Jebel Ali Freezone, Dubai."},
    {"id": "en_loc_visit", "text": "Customers can visit the Jebel Ali warehouse to pick up items directly."},
    {"id": "en_policy_check_before_pay", "text": "Customers have the right to open the box and check the phone before making the payment."},
    {"id": "en_policy_replacement", "text": "We provide a 7-day replacement warranty for any technical issues."},
    {"id": "en_policy_cod", "text": "Cash on Delivery (COD) is the standard payment method; check the item first, then pay."},
    {"id": "en_policy_order_req", "text": "To process an order, we require your Google Maps location and a valid calling number."},
    {"id": "en_variant_redmi_14c", "text": "Redmi 14c is available in 5G variants."},
    {"id": "en_variant_s25_ultra_offer", "text": "The S25 Ultra 210 AED offer is strictly limited to 24 hours."},
    {"id": "en_variant_colors", "text": "Stock availability includes various colors; please ask for current color photos on WhatsApp."},
    {"id": "en_promo_1", "text": "Offer is just for 24 hours - S25 Ultra for 210 AED."},
    {"id": "en_promo_2", "text": "Seal pack box pack phone available now."},
    {"id": "en_promo_3", "text": "Pay only after you check the phone."},
    {"id": "en_promo_4", "text": "Wholesale prices available for bulk orders of 2 or more pieces."},
    {"id": "en_order_process", "text": "To place an order, please provide your Full Name, Mobile Number, and Google Maps Location."},
    {"id": "en_order_timing", "text": "Orders confirmed before 4 PM are processed for next-day delivery dispatch."},
    {"id": "en_order_confirmation", "text": "Our team will call you to verify the order before the rider departs."},

    # --- Urdu ---
    {"id": "ur_order_process_1", "text": "آرڈر پلیس کرنے کا طریقہ یہ ہے کہ آپ اپنا پورا نام، موبائل نمبر اور لوکیشن سینڈ کریں۔"},
    {"id": "ur_delivery_time_1", "text": "متحدہ عرب امارات میں آرڈر 2 سے 3 دن کے اندر ڈیلیور ہوتا ہے۔ کل آپ کو مل جائے گا۔"},
    {"id": "ur_cod_1", "text": "کیش آن ڈیلیوری پورے UAE میں دستیاب ہے۔ آپ فون چیک کرنے کے بعد رقم ادا کر سکتے ہیں۔"},
    {"id": "ur_warranty_1", "text": "تمام فونز 7 دن کی ریپلیسمنٹ وارنٹی کے ساتھ آتے ہیں۔"},
    {"id": "ur_contact_1", "text": "مزید معلومات کے لیے ہم سے رابطہ کریں یا 0563235075 پر کال کریں۔"},
    {"id": "ur_price_s25_ultra_1", "text": "ایس 25 الٹرا کی قیمت 24 گھنٹے کے آفر میں صرف 210 درہم ہے۔"},
    {"id": "ur_price_s24_ultra_1", "text": "سیمسنگ ایس 24 الٹرا (ہائی کوالٹی کاپی) 375 درہم میں دستیاب ہے۔"},
    {"id": "ur_price_redmi_1", "text": "ریڈمی 14C کی قیمت 210 درہم ہے۔"},
    {"id": "ur_spec_condition_1", "text": "تمام فون نئے، سیل پیک اور اصل باکس کے ساتھ آتے ہیں۔"},
    {"id": "ur_spec_contents_1", "text": "باکس میں چارجر اور دیگر ضروری لوازمات شامل ہوتے ہیں۔"},
    {"id": "ur_delivery_area_1", "text": "ہم دبئی، عجمان اور دیگر تمام علاقوں میں ہوم ڈیلیوری فراہم کرتے ہیں۔"},
    {"id": "ur_delivery_charges_1", "text": "ڈیلیوری چارجز عام طور پر 10 درہم ہوتے ہیں۔"},
    {"id": "ur_delivery_charges_free", "text": "کچھ آفرز میں ڈیلیوری فری بھی دی جاتی ہے۔"},
    {"id": "ur_delivery_process_1", "text": "رائیڈر آنے سے پہلے آپ سے فون پر رابطہ کرے گا۔"},
    {"id": "ur_visit_1", "text": "آپ خود آ کر بھی ہمارا وئیر ہاؤس وزٹ کر سکتے ہیں۔"},
    {"id": "ur_policy_check_1", "text": "پہلے پراڈکٹ دیکھیں پھر پیمنٹ کریں۔"},
    {"id": "ur_promo_limited", "text": "یہ محدود وقت کی آفر ہے، جلدی آرڈر کریں۔"},
    {"id": "ur_order_timing_1", "text": "شام 4 بجے سے پہلے کنفرم ہونے والے آرڈرز اگلے دن بھیجے جاتے ہیں۔"},

    # --- Urdu Queries ---
    {"id": "ur_q_price_1", "text": "اس فون کی قیمت کیا ہے؟"},
    {"id": "ur_q_delivery_1", "text": "ڈیلیوری کتنے دن میں ہوگی؟"},
    {"id": "ur_q_cod_1", "text": "کیا کیش آن ڈیلیوری دستیاب ہے؟"},
    {"id": "ur_q_original_1", "text": "کیا یہ فون اوریجنل ہے؟"},
    {"id": "ur_q_order_1", "text": "آرڈر کیسے کرنا ہے؟"},
    {"id": "ur_q_warranty_1", "text": "کیا وارنٹی ملتی ہے؟"},
    {"id": "ur_q_stock_1", "text": "کیا یہ ابھی دستیاب ہے؟"},
    {"id": "ur_q_location_1", "text": "آپ کہاں واقع ہیں؟"},

    # --- Roman Urdu ---
    {"id": "ru_order_process_1", "text": "Order place karne ka tareeqa yeh hai ke aap apna pura naam, mobile number aur location send karein."},
    {"id": "ru_delivery_time_1", "text": "UAE mein order 2 se 3 din mein deliver hota hai. Kal aapko mil jayega."},
    {"id": "ru_cod_1", "text": "Cash on Delivery poore UAE mein available hai. Aap phone check karne ke baad payment kar sakte hain."},
    {"id": "ru_warranty_1", "text": "Sab phones 7 din ki replacement warranty ke saath aate hain."},
    {"id": "ru_contact_1", "text": "Zyada information ke liye humein WhatsApp karein ya 0563235075 par call karein."},
    {"id": "ru_price_s25_ultra_1", "text": "S25 Ultra ki price 24 hours offer mein sirf 210 AED hai."},
    {"id": "ru_price_s24_ultra_1", "text": "Samsung S24 Ultra (High Quality Copy) 375 AED mein available hai."},
    {"id": "ru_price_redmi_1", "text": "Redmi 14C ki price 210 AED hai."},
    {"id": "ru_price_s23_ultra_1", "text": "S23 Ultra ki price 210 AED hai."},
    {"id": "ru_order_tracking_1", "text": "aap apna order phone number ya order id se track kar sakte hain."},
    {"id": "ru_spec_fingerprint_1", "text": "phones mein fingerprint sensor screen ke andar ya side par hota hai."},
    {"id": "ru_spec_condition_1", "text": "tamam phones bilkul naye hain, seal pack aur original box ke sath aate hain."},
    {"id": "ru_delivery_area_1", "text": "hum UAE ke tamam shehron mein delivery karte hain jaise Dubai, Ajman aur Jebel Ali."},
    {"id": "ru_delivery_charges_1", "text": "delivery charges aam tor par 10 AED hain lekin kuch offers mein free bhi hoti hai."},
    {"id": "ru_location_ajman_1", "text": "Ags Gadgets ka base Ajman mein hai."},
    {"id": "ru_location_jebel_ali_1", "text": "hamara warehouse Jebel Ali Freezone Dubai mein hai."},
    {"id": "ru_policy_check_1", "text": "aap payment se pehle box open karke phone check kar sakte hain."},
    {"id": "ru_variant_colors_1", "text": "different colors available hain, WhatsApp par photos mang sakte hain."},
    {"id": "ru_promo_bulk", "text": "2 ya us se zyada phones lene par wholesale price milega."},
    {"id": "ru_order_timing_1", "text": "4 baje se pehle confirm hone wale orders next day dispatch hote hain."},

    # --- Roman Urdu Queries ---
    {"id": "ru_q_price_1", "text": "price kya hai?"},
    {"id": "ru_q_price_2", "text": "yeh kitne ka hai?"},
    {"id": "ru_q_price_3", "text": "S25 Ultra ki price kya hai?"},

    {"id": "ru_q_delivery_1", "text": "delivery kitne din mein hogi?"},
    {"id": "ru_q_delivery_2", "text": "kal mil jayega?"},
    {"id": "ru_q_delivery_3", "text": "Dubai delivery hai?"},

    {"id": "ru_q_cod_1", "text": "COD available hai?"},
    {"id": "ru_q_cod_2", "text": "check karke pay kar sakte hain?"},
    {"id": "ru_q_cod_3", "text": "pehle payment karni hogi?"},

    {"id": "ru_q_original_1", "text": "original hai ya copy?"},
    {"id": "ru_q_original_2", "text": "new hai ya used?"},

    {"id": "ru_q_order_1", "text": "order kaise karein?"},
    {"id": "ru_q_order_2", "text": "order place karna hai"},

    {"id": "ru_q_warranty_1", "text": "warranty hai?"},
    {"id": "ru_q_warranty_2", "text": "issue ho gaya to kya hoga?"},

    {"id": "ru_q_stock_1", "text": "available hai?"},
    {"id": "ru_q_stock_2", "text": "stock mein hai?"},

    {"id": "ru_q_location_1", "text": "aap kahan ho?"},
    {"id": "ru_q_location_2", "text": "shop kahan hai?"},

    {"id": "ru_q_contact_1", "text": "WhatsApp number bhejo"},
    {"id": "ru_q_offer_1", "text": "offer abhi available hai?"},

    # --- Roman Urdu Edge Cases ---
    {"id": "ru_edge_1", "text": "agar delivery time par main available na hua to?"},
    {"id": "ru_edge_2", "text": "phone check karne ke baad pasand na aaye to?"},
    {"id": "ru_edge_3", "text": "return policy kya hai?"},
    {"id": "ru_edge_4", "text": "UAE ke bahar delivery hai?"},
    {"id": "ru_edge_5", "text": "agar number reachable na ho to?"},
    {"id": "ru_edge_6", "text": "koi aur receive kar sakta hai?"},
    {"id": "ru_edge_7", "text": "online payment accept karte ho?"},
    {"id": "ru_edge_8", "text": "agar product damaged nikle to?"},
    {"id": "ru_edge_9", "text": "koi hidden charges to nahi?"},
    {"id": "ru_edge_10", "text": "order cancel kar sakta hoon?"},

    # --- Noisy / Real Typing Roman Urdu ---
    {"id": "ru_noise_1", "text": "price btao"},
    {"id": "ru_noise_2", "text": "kitna last?"},
    {"id": "ru_noise_3", "text": "final price kya hai"},
    {"id": "ru_noise_4", "text": "location send kro"},
    {"id": "ru_noise_5", "text": "COD h?"},
    {"id": "ru_noise_6", "text": "available h khtm hogya?"},
    {"id": "ru_noise_7", "text": "kal tak mil skta?"},
    {"id": "ru_noise_8", "text": "ye original h na?"},
    {"id": "ru_noise_9", "text": "order krna h"},
    {"id": "ru_noise_10", "text": "msg ka reply q nai ara?"},
]

# 4. Uploading to Database
print(f"📥 Adding {len(documents)} documents to the knowledge base...")

for doc in documents:
    embedding = embedder.encode(doc["text"]).tolist()
    collection.add(
        ids=[doc["id"]],
        embeddings=[embedding],
        documents=[doc["text"]]
    )

print("✅ Knowledge Base Updated Successfully!")