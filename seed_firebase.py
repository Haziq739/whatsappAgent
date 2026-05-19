import os
from utils.firebase_db import db
import uuid

def seed_products():
    if not db:
        print("❌ Firebase not initialized. Exiting.")
        return

    products_ref = db.collection('products')
    
    # Check if products already exist to avoid duplication
    existing = products_ref.limit(1).get()
    if existing:
        print("⚠️ Products already exist in Firebase. Clear the collection first if you want to re-seed.")
        return

    products_data = [
        {"name": "Redmi 14c", "price": 159, "stock_status": "in_stock"},
        {"name": "Redmi 14c 5G", "price": 210, "stock_status": "in_stock"},
        {"name": "Samsung Galaxy S25 Ultra", "price": 210, "stock_status": "in_stock"},
        {"name": "Samsung Galaxy S24 Ultra (High Quality Copy)", "price": 375, "stock_status": "in_stock"},
        {"name": "Samsung Galaxy S23 Ultra", "price": 180, "stock_status": "in_stock"},
        {"name": "Samsung Galaxy A15", "price": 125, "stock_status": "in_stock"},
        {"name": "Samsung Galaxy A25", "price": 165, "stock_status": "in_stock"},
        {"name": "Samsung Galaxy A35", "price": 220, "stock_status": "in_stock"},
        {"name": "Samsung Galaxy A55", "price": 285, "stock_status": "in_stock"},
        {"name": "Samsung Galaxy A05s", "price": 99, "stock_status": "in_stock"},
        {"name": "Samsung Galaxy A06", "price": 110, "stock_status": "in_stock"},
        {"name": "Samsung Galaxy Z Fold 6", "price": 1450, "stock_status": "in_stock"},
        {"name": "Samsung Galaxy Z Flip 6", "price": 950, "stock_status": "in_stock"},
        {"name": "Samsung Galaxy Z Fold 5", "price": 1150, "stock_status": "in_stock"},
        {"name": "Samsung Galaxy S24 FE", "price": 350, "stock_status": "in_stock"},
        {"name": "Samsung Galaxy S24 Plus", "price": 420, "stock_status": "in_stock"},
        {"name": "Samsung Galaxy S24 (Standard)", "price": 310, "stock_status": "in_stock"},
        
        {"name": "iPhone 16 Pro Max", "price": 1650, "stock_status": "in_stock"},
        {"name": "iPhone 16 Pro", "price": 1450, "stock_status": "in_stock"},
        {"name": "iPhone 16", "price": 1150, "stock_status": "in_stock"},
        {"name": "iPhone 15 Pro Max", "price": 1350, "stock_status": "in_stock"},
        {"name": "iPhone 15 Pro", "price": 1200, "stock_status": "in_stock"},
        {"name": "iPhone 15", "price": 950, "stock_status": "in_stock"},
        {"name": "iPhone 14 Pro Max", "price": 1100, "stock_status": "in_stock"},
        {"name": "iPhone 14 Pro", "price": 980, "stock_status": "in_stock"},
        {"name": "iPhone 14", "price": 750, "stock_status": "in_stock"},
        {"name": "iPhone 13", "price": 620, "stock_status": "in_stock"},
        {"name": "iPhone 12", "price": 480, "stock_status": "in_stock"},
        {"name": "iPhone 11", "price": 350, "stock_status": "in_stock"},
        {"name": "iPhone SE (2022)", "price": 420, "stock_status": "in_stock"},
        
        {"name": "Google Pixel 8 Pro", "price": 850, "stock_status": "in_stock"},
        {"name": "Google Pixel 8", "price": 650, "stock_status": "in_stock"},
        {"name": "Google Pixel 8a", "price": 520, "stock_status": "in_stock"},
        {"name": "Google Pixel 7a", "price": 420, "stock_status": "in_stock"},
        {"name": "Google Pixel 7", "price": 380, "stock_status": "in_stock"},
        {"name": "Google Pixel 6", "price": 290, "stock_status": "in_stock"},
        
        {"name": "Redmi Note 13", "price": 195, "stock_status": "in_stock"},
        {"name": "Redmi Note 13 Pro", "price": 265, "stock_status": "in_stock"},
        {"name": "Redmi Note 12", "price": 175, "stock_status": "in_stock"},
        {"name": "POCO X6 Pro", "price": 310, "stock_status": "in_stock"},
        {"name": "POCO M6 Pro", "price": 220, "stock_status": "in_stock"},
        {"name": "POCO F6", "price": 380, "stock_status": "in_stock"},
        {"name": "Xiaomi 14", "price": 550, "stock_status": "in_stock"},
        {"name": "Xiaomi 14 Ultra", "price": 750, "stock_status": "in_stock"},
        
        {"name": "MacBook Air M1 (2020)", "price": 1200, "stock_status": "in_stock"},
        {"name": "MacBook Air M2", "price": 1550, "stock_status": "in_stock"},
        {"name": "MacBook Air M3", "price": 1850, "stock_status": "in_stock"},
        {"name": "MacBook Pro M2 14-inch", "price": 2200, "stock_status": "in_stock"},
        {"name": "MacBook Pro M3 14-inch", "price": 2650, "stock_status": "in_stock"},
        {"name": "MacBook Pro M3 Pro 16-inch", "price": 3200, "stock_status": "in_stock"},
        
        {"name": "HP EliteBook 840 G9", "price": 1450, "stock_status": "in_stock"},
        {"name": "HP Pavilion 15", "price": 850, "stock_status": "in_stock"},
        {"name": "HP OMEN 16 Gaming Laptop", "price": 1950, "stock_status": "in_stock"},
        {"name": "HP 14s Laptop", "price": 650, "stock_status": "in_stock"},
        {"name": "HP Victus 15 Gaming Laptop", "price": 1350, "stock_status": "in_stock"},
        
        {"name": "Dell Latitude 7490", "price": 1100, "stock_status": "in_stock"},
        {"name": "Dell Inspiron 15", "price": 780, "stock_status": "in_stock"},
        {"name": "Dell XPS 13", "price": 1650, "stock_status": "in_stock"},
        {"name": "Dell XPS 15", "price": 2100, "stock_status": "in_stock"},
        {"name": "Dell G15 Gaming Laptop", "price": 1500, "stock_status": "in_stock"},
        
        {"name": "Lenovo ThinkPad X1 Carbon", "price": 1750, "stock_status": "in_stock"},
        {"name": "Lenovo IdeaPad 3", "price": 620, "stock_status": "in_stock"},
        {"name": "Lenovo Legion 5 Gaming Laptop", "price": 1680, "stock_status": "in_stock"},
        {"name": "Lenovo Yoga 7 2-in-1", "price": 1200, "stock_status": "in_stock"},
        
        {"name": "ASUS ROG Strix G16 Gaming Laptop", "price": 2100, "stock_status": "in_stock"},
        {"name": "ASUS TUF Gaming F15 Laptop", "price": 1400, "stock_status": "in_stock"},
        {"name": "ASUS VivoBook 15", "price": 680, "stock_status": "in_stock"},
        {"name": "ASUS ZenBook 14 OLED", "price": 1350, "stock_status": "in_stock"}
    ]

    print("🔄 Seeding products into Firebase...")
    batch = db.batch()
    
    count = 0
    for product in products_data:
        doc_ref = products_ref.document()
        # Add id for convenience matching later
        product["id"] = doc_ref.id
        batch.set(doc_ref, product)
        count += 1
        
        # Firestore batch has a limit of 500, we have much less, but good practice
        if count % 400 == 0:
            batch.commit()
            batch = db.batch()

    if count % 400 != 0:
        batch.commit()
        
    print(f"✅ Successfully seeded {count} products!")

if __name__ == "__main__":
    seed_products()
