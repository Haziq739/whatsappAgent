from flask import Flask, request, jsonify
from pyngrok import ngrok
import requests
from langdetect import detect
from utils.transcription import transcribe_audio
from utils.brain import generate_rag_response
from utils.whatsapp_api import send_whatsapp_text, send_whatsapp_audio
# ✅ IMPORTED the TTS engine you just made
from utils.tts_engine import generate_audio_sync 
from utils.firebase_db import store_message, create_order
import config

app = Flask(__name__)

# Prevent duplicate message handling
processed_message_ids = set()

@app.route("/webhook", methods=["GET", "POST"])
def webhook():
    if request.method == "GET":
        # ✅ Webhook verification (Must match your Meta Developer Portal)
        verify_token = "mytoken123" 
        mode = request.args.get("hub.mode")
        token = request.args.get("hub.verify_token")
        challenge = request.args.get("hub.challenge")

        if mode == "subscribe" and token == verify_token:
            print("✅ Webhook verified successfully!")
            return challenge, 200
        return "Forbidden", 403

    elif request.method == "POST":
        data = request.get_json()
        
        try:
            entry = data.get("entry", [])[0]
            changes = entry.get("changes", [])[0]
            value = changes.get("value", {})

            # ✅ Ignore status updates (delivered/read)
            if "statuses" in value:
                return jsonify({"status": "ignored_status"}), 200

            messages = value.get("messages", [])
            if not messages:
                return jsonify({"status": "no_message"}), 200

            message_obj = messages[0]
            sender = message_obj.get("from")

            # ✅ Prevent duplicate message handling
            message_id = message_obj.get("id")
            if message_id in processed_message_ids:
                return jsonify({"status": "duplicate_message"}), 200
            processed_message_ids.add(message_id)

            # --- HANDLE TEXT MESSAGES ---
            if "text" in message_obj:
                user_text = message_obj["text"].get("body", "").strip()
                agent_number = getattr(config, 'PHONE_NUMBER_ID', 'Agent')
                
                print(f"[RECEIVED] From: {sender} → To: {agent_number}")
                print(f"Message: {user_text}\n")
                
                # Store incoming text in Firebase
                store_message(sender, agent_number, user_text, "customer")
                
                # Generate response from AI + Knowledge Base
                response_text, user_lang = generate_rag_response(user_text, sender=sender)
                
                # Check if order was confirmed
                from utils.brain import CONFIRMED_ORDERS_QUEUE
                if sender in CONFIRMED_ORDERS_QUEUE:
                    order_info = CONFIRMED_ORDERS_QUEUE.pop(sender)
                    create_order(sender, order_info["details"], order_info["product"])
                
                print(f"[SENT] From: {agent_number} → To: {sender}")
                print(f"Message: {response_text}\n")
                
                # Store outgoing text in Firebase
                store_message(sender, agent_number, response_text, "agent")

                send_whatsapp_text(sender, response_text)

            # --- HANDLE VOICE/AUDIO MESSAGES ---
            elif "audio" in message_obj:
                print("🎧 Received voice message")
                audio_id = message_obj["audio"]["id"]

                # Step 1: Get media download URL from Meta
                media_url = f"https://graph.facebook.com/v17.0/{audio_id}"
                headers = {"Authorization": f"Bearer {config.ACCESS_TOKEN}"}
                
                r = requests.get(media_url, headers=headers).json()
                download_url = r.get("url")

                if download_url:
                    # Step 2: Download the .ogg file
                    audio_bytes = requests.get(download_url, headers=headers).content
                    audio_path = "voice.ogg"
                    with open(audio_path, "wb") as f:
                        f.write(audio_bytes)

                    # Step 3: Speech-to-Text (Transcription)
                    # This uses the ffmpeg.exe you placed in the folder
                    text_input = transcribe_audio(audio_path)
                    agent_number = getattr(config, 'PHONE_NUMBER_ID', 'Agent')
                    
                    print(f"[RECEIVED] From: {sender} → To: {agent_number}")
                    print(f"Message (Audio Transcribed): {text_input}\n")
                    
                    # Store incoming audio transcribed text
                    store_message(sender, agent_number, f"[Audio] {text_input}", "customer")

                    # Step 4 & 5: Generate AI Response (Language detection now handled inside brain.py)
                    ai_response, tts_lang = generate_rag_response(text_input, sender=sender)
                    
                    # ✅ SPECIAL CASE: If we are at the order confirmation step, send TEXT instead of audio
                    from utils.brain import USER_STATES
                    is_confirmation_step = sender in USER_STATES and USER_STATES[sender].get("step") == "awaiting_confirmation"

                    # Check if order was confirmed
                    from utils.brain import CONFIRMED_ORDERS_QUEUE
                    if sender in CONFIRMED_ORDERS_QUEUE:
                        order_info = CONFIRMED_ORDERS_QUEUE.pop(sender)
                        create_order(sender, order_info["details"], order_info["product"])

                    if is_confirmation_step:
                        print(f"[SENT] From: {agent_number} → To: {sender} (Text Confirmation)")
                        print(f"Message (Text): {ai_response}\n")
                        send_whatsapp_text(sender, ai_response)
                        store_message(sender, agent_number, ai_response, "agent")
                        return jsonify({"status": "ok"}), 200

                    print(f"[SENT] From: {agent_number} → To: {sender}")
                    print(f"Message (Audio): {ai_response}\n")
                    
                    # Store outgoing generated text
                    store_message(sender, agent_number, f"[Audio Reply] {ai_response}", "agent")

                    # Step 6: Convert AI Response to Audio and Send
                    
                    print(f"🎙️ Generating voice reply in {tts_lang}...")
                    
                    # ✅ ACTUALLY GENERATING THE VOICE NOW
                    audio_reply_path = generate_audio_sync(ai_response, lang=tts_lang)
                    
                    # ✅ SENDING THE VOICE BACK TO WHATSAPP
                    send_whatsapp_audio(sender, audio_reply_path)
                    
                    print(f"✅ Voice reply sent in {tts_lang}")

                else:
                    print("❌ Error: Could not get audio download URL")

            else:
                print(f"⚠️ Unsupported message type: {message_obj.keys()}")

        except Exception as e:
            print("❌ Error processing message:", e)

        return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    # Setup Ngrok
    ngrok.set_auth_token(config.NGROK_AUTH_TOKEN)
    public_url = ngrok.connect(5000)
    
    print(f"🚀 Ags Gadgets Bot is LIVE!")
    print(f"🌍 PUBLIC URL: {public_url}/webhook")
    print("📋 Copy the URL above and paste it into Meta Developer Portal.")
    
    app.run(port=5000)