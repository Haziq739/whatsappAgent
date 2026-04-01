import requests
import os
import config

def send_whatsapp_text(to, message):
    """Sends a text message via WhatsApp Business API"""
    url = f"https://graph.facebook.com/v17.0/{config.PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {config.ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": message},
    }
    try:
        res = requests.post(url, json=data, headers=headers)
        print(f"📤 WhatsApp Text Status: {res.status_code}")
        return res.json()
    except Exception as e:
        print(f"❌ Error sending WhatsApp text: {e}")
        return None

def send_whatsapp_audio(to, audio_path):
    """Uploads and sends an audio file via WhatsApp Business API"""
    try:
        # 1. Upload the media
        url_upload = f"https://graph.facebook.com/v17.0/{config.PHONE_NUMBER_ID}/media"
        headers = {"Authorization": f"Bearer {config.ACCESS_TOKEN}"}
        
        files = {
            "file": (os.path.basename(audio_path), open(audio_path, "rb"), "audio/mpeg")
        }
        data = {"messaging_product": "whatsapp"}
        
        upload_res = requests.post(url_upload, headers=headers, files=files, data=data)
        media_id = upload_res.json().get("id")

        if not media_id:
            print("❌ Failed to get Media ID from WhatsApp")
            return

        # 2. Send the message using the Media ID
        url_send = f"https://graph.facebook.com/v17.0/{config.PHONE_NUMBER_ID}/messages"
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "audio",
            "audio": {"id": media_id},
        }
        
        res = requests.post(url_send, headers=headers, json=payload)
        print(f"📤 WhatsApp Audio Status: {res.status_code}")
        return res.json()
    except Exception as e:
        print(f"❌ Error sending WhatsApp audio: {e}")