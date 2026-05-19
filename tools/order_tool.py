from utils.firebase_db import create_order

def order_creation_tool(customer_number, order_info, product_name=None):
    """
    Wrapper tool to submit orders securely to Firebase.
    order_info can be a dict with keys: details, customer_name, customer_phone, customer_location
    or a plain string (legacy fallback).
    """
    # Support both old-style (string) and new-style (dict) calls
    if isinstance(order_info, dict):
        p_name = product_name or order_info.get("product", "Unknown Product")
        c_name = order_info.get("customer_name", "Customer")
        c_phone = order_info.get("customer_phone", "")
        c_location = order_info.get("customer_location", "")
        raw_details = order_info.get("details", "")
    else:
        p_name = product_name or "Unknown Product"
        c_name = ""
        c_phone = ""
        c_location = ""
        raw_details = order_info

    print(f"[TOOL CALLED] OrderTool -> Creating order for: '{p_name}'")
    print(f"[TOOL INFO] Name: {c_name}, Phone: {c_phone}, Location: {c_location}")
    try:
        create_order(
            customer_number=customer_number,
            raw_details=raw_details,
            product_name=p_name,
            customer_name=c_name,
            customer_phone=c_phone,
            customer_location=c_location
        )
        print(f"[TOOL RESULT] Order created successfully.")
        return True
    except Exception as e:
        print(f"[TOOL ERROR] Failed to create order: {e}")
        return False
