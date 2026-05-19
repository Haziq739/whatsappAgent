import chromadb
from sentence_transformers import SentenceTransformer

# 1. Initialize the embedding model (Same one used in brain.py)
print("Initializing embedding engine...")
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# 2. Connect to the database folder
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="business_Knowledge")

# 3. Your Business Data (ONLY FAQ & Policies)
documents = [
    # --- English User Queries (High Priority) ---
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

    {"id": "en_q_warranty_1", "text": "Do you give warranty?"},
    {"id": "en_q_warranty_2", "text": "What if the phone has an issue?"},
    {"id": "en_q_warranty_3", "text": "Replacement policy?"},

    {"id": "en_q_location_1", "text": "Where are you located?"},
    {"id": "en_q_location_2", "text": "Can I visit your shop?"},
    {"id": "en_q_location_3", "text": "Do you have a physical store?"},

    {"id": "en_q_contact_1", "text": "Your WhatsApp number?"},
    {"id": "en_q_contact_2", "text": "How can I contact you?"},

    # --- Edge Cases ---
    {"id": "en_edge_2", "text": "What if I don't like the phone after checking?"},
    {"id": "en_edge_3", "text": "Can I return the phone?"},
    {"id": "en_edge_4", "text": "Do you deliver outside UAE?"},
    {"id": "en_edge_10", "text": "Can I cancel my order?"},

    {"id": "en_1", "text": "We deliver all orders within 2 to 3 working days."},
    {"id": "en_2", "text": "You can track your order using your phone number or order ID."},
    {"id": "en_3", "text": "Cash on Delivery is available across the UAE."},
    {"id": "en_4", "text": "Our phones come with a 7-day replacement warranty."},
    {"id": "en_5", "text": "For more information, contact us on WhatsApp or call 0563235075."},
    {"id": "en_spec_fingerprint", "text": "The phones feature a fingerprint sensor located inside/on the screen."},
    {"id": "en_spec_condition", "text": "All phones are brand new, seal-packed, and come in the original box pack."},
    {"id": "en_spec_originality", "text": "The gadgets are described as original quality products."},
    {"id": "en_spec_contents", "text": "The phone box includes a charger and all standard accessories."},
    {"id": "en_delivery_area", "text": "We provide home delivery across the UAE, including Dubai, Ajman, and Jebel Ali."},
    {"id": "en_delivery_time_standard", "text": "Standard delivery takes 2 to 3 working days."},
    {"id": "en_delivery_time_fast", "text": "Next-day delivery is available for orders confirmed before the cutoff time."},
    {"id": "en_delivery_charges", "text": "Delivery charges are typically 10 AED, though free delivery is offered on specific deals."},
    {"id": "en_delivery_process", "text": "The rider will contact you via phone call before arriving at your location."},
    {"id": "en_loc_ajman", "text": "Ags Gadgets has a presence in Ajman."},
    {"id": "en_loc_jebel_ali", "text": "Our warehouse is located in Jebel Ali Freezone, Dubai."},
    {"id": "en_loc_visit", "text": "Customers can visit the Jebel Ali warehouse to pick up items directly."},
    {"id": "en_policy_check_before_pay", "text": "Customers have the right to open the box and check the phone before making the payment."},
    {"id": "en_policy_replacement", "text": "We provide a 7-day replacement warranty for any technical issues."},
    {"id": "en_policy_cod", "text": "Cash on Delivery (COD) is the standard payment method; check the item first, then pay."},
    {"id": "en_policy_order_req", "text": "To process an order, we require your Full Name, Mobile Number, and Google Maps Location."},
    {"id": "en_promo_3", "text": "Pay only after you check the phone."},
    {"id": "en_promo_4", "text": "Wholesale prices available for bulk orders of 2 or more pieces."},
    {"id": "en_order_process", "text": "To place an order, please provide your Full Name, Mobile Number, and Google Maps Location."},
    {"id": "en_order_timing", "text": "Orders confirmed before 4 PM are processed for next-day delivery dispatch."},

    # --- Urdu ---
    {"id": "ur_order_process_1", "text": "آرڈر پلیس کرنے کا طریقہ یہ ہے کہ آپ اپنا پورا نام، موبائل نمبر اور لوکیشن سینڈ کریں۔"},
    {"id": "ur_delivery_time_1", "text": "متحدہ عرب امارات میں آرڈر 2 سے 3 دن کے اندر ڈیلیور ہوتا ہے۔"},
    {"id": "ur_cod_1", "text": "کیش آن ڈیلیوری پورے UAE میں دستیاب ہے۔ آپ فون چیک کرنے کے بعد رقم ادا کر سکتے ہیں۔"},
    {"id": "ur_warranty_1", "text": "تمام فونز 7 دن کی ریپلیسمنٹ وارنٹی کے ساتھ آتے ہیں۔"},
    {"id": "ur_contact_1", "text": "مزید معلومات کے لیے ہم سے رابطہ کریں یا 0563235075 پر کال کریں۔"},
    {"id": "ur_spec_condition_1", "text": "تمام فون نئے، سیل پیک اور اصل باکس کے ساتھ آتے ہیں۔"},
    {"id": "ur_spec_contents_1", "text": "باکس میں چارجر اور دیگر ضروری لوازمات شامل ہوتے ہیں۔"},
    {"id": "ur_delivery_area_1", "text": "ہم دبئی، عجمان اور دیگر تمام علاقوں میں ہوم ڈیلیوری فراہم کرتے ہیں۔"},
    {"id": "ur_delivery_charges_1", "text": "ڈیلیوری چارجز عام طور پر 10 درہم ہوتے ہیں۔"},
    {"id": "ur_delivery_process_1", "text": "رائیڈر آنے سے پہلے آپ سے فون پر رابطہ کرے گا۔"},
    {"id": "ur_visit_1", "text": "آپ خود آ کر بھی ہمارا وئیر ہاؤس وزٹ کر سکتے ہیں۔"},
    {"id": "ur_policy_check_1", "text": "پہلے پراڈکٹ دیکھیں پھر پیمنٹ کریں۔"},

    # --- Roman Urdu ---
    {"id": "ru_order_process_1", "text": "Order place karne ka tareeqa yeh hai ke aap apna pura naam, mobile number aur location send karein."},
    {"id": "ru_delivery_time_1", "text": "UAE mein order 2 se 3 din mein deliver hota hai."},
    {"id": "ru_cod_1", "text": "Cash on Delivery poore UAE mein available hai. Aap phone check karne ke baad payment kar sakte hain."},
    {"id": "ru_warranty_1", "text": "Sab phones 7 din ki replacement warranty ke saath aate hain."},
    {"id": "ru_contact_1", "text": "Zyada information ke liye humein WhatsApp karein ya 0563235075 par call karein."},
    {"id": "ru_spec_fingerprint_1", "text": "phones mein fingerprint sensor screen ke andar ya side par hota hai."},
    {"id": "ru_spec_condition_1", "text": "tamam phones bilkul naye hain, seal pack aur original box ke sath aate hain."},
    {"id": "ru_delivery_area_1", "text": "hum UAE ke tamam shehron mein delivery karte hain jaise Dubai, Ajman aur Jebel Ali."},
    {"id": "ru_delivery_charges_1", "text": "delivery charges aam tor par 10 AED hain."},
    {"id": "ru_location_ajman_1", "text": "Ags Gadgets ka base Ajman mein hai."},
    {"id": "ru_location_jebel_ali_1", "text": "hamara warehouse Jebel Ali Freezone Dubai mein hai."},
    {"id": "ru_policy_check_1", "text": "aap payment se pehle box open karke phone check kar sakte hain."},

    # --- Additional Edge Case FAQ Entries ---
    {"id": "en_edge_unavailable", "text": "If you are not available at the time of delivery, the rider will call you to reschedule."},
    {"id": "en_edge_someone_else", "text": "Yes, someone else can receive your order on your behalf. Just inform us of their name and number."},
    {"id": "en_edge_not_reachable", "text": "If your number is not reachable, the rider will try again later or reschedule the delivery."},
    {"id": "en_edge_outside_uae", "text": "Currently, we deliver only within the UAE. International shipping is not available at this time."},
    {"id": "en_edge_advance_pay", "text": "No advance payment is required. We offer Cash on Delivery (COD) — pay only after checking the phone."},
    {"id": "en_edge_cancel_order", "text": "Yes, you can cancel your order anytime before it is dispatched by contacting us on WhatsApp."},
    {"id": "en_edge_return_phone", "text": "If you don't like the phone after checking, you can refuse the delivery. We also offer 7-day replacement for defects."},
    {"id": "en_edge_cod_cities", "text": "Cash on delivery (COD) is available in all cities across the UAE."},
    {"id": "en_edge_nextday", "text": "Yes, we offer next-day delivery for orders confirmed before 4 PM."},
    {"id": "en_edge_delivery_charge", "text": "Delivery charges are typically 10 AED. Free delivery is available on select deals."},
    {"id": "en_edge_whatsapp_num", "text": "You can contact us on WhatsApp at 0563235075."},
    {"id": "en_edge_physical_store", "text": "Yes, we have a warehouse in Jebel Ali Freezone, Dubai. You can visit to pick up items directly."},
    {"id": "en_edge_box_contents", "text": "All phones come in the original box pack with a charger and standard accessories included."},
    {"id": "en_edge_phone_condition", "text": "All our phones are brand new, seal-packed, and 100% original quality."},
    {"id": "en_edge_order_details", "text": "To place an order, we need your Full Name, Mobile Number, and Google Maps Location."},
    {"id": "en_edge_ordering_process", "text": "The ordering process is simple: share your details (name, number, location), we confirm your order, and deliver within 2-3 days with Cash on Delivery."},
    {"id": "en_edge_available_products", "text": "We have a wide range of mobile phones available, including popular brands like Samsung, iPhone, Vivo, Xiaomi, Redmi, Nokia, and Tecno. We also sell iPads, laptops, airpods, and smartwatches. Please specify a brand or model you are looking for to check availability and price."},
    {"id": "ur_edge_available_products", "text": "ہمارے پاس سام سنگ، آئی فون، ویوو، شومی، اور دیگر مشہور برانڈز کے موبائل فونز دستیاب ہیں۔ اس کے علاوہ لیپ ٹاپس، آئی پیڈز اور سمارٹ واچز بھی ہیں۔ براہ کرم اپنی پسند کا برانڈ بتائیں۔"},
    {"id": "ru_edge_available_products", "text": "Hamare paas Samsung, iPhone, Vivo, Xiaomi aur dosre popular brands ke mobile phones available hain. Laptops, iPads aur earbuds bhi hain. Aap ko konsa brand chahiye?"},
]

# 4. Uploading to Database
print(f"Adding {len(documents)} documents to the knowledge base...")

for doc in documents:
    embedding = embedder.encode(doc["text"]).tolist()
    collection.add(
        ids=[doc["id"]],
        embeddings=[embedding],
        documents=[doc["text"]]
    )

print("Knowledge Base Updated Successfully!")