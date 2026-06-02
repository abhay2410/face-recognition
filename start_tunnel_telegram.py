import os
import re
import subprocess
import time
import urllib.request
import urllib.parse
import json

# Configuration
BOT_TOKEN = "8732799919:AAFJhmY2hgXHaNZXjwGindO8kaUNVLibPL4"
CLOUDFLARED_PATH = r"C:\Program Files (x86)\cloudflared\cloudflared.exe"
PORT = 8000
ENV_FILE = ".env"

def load_or_get_chat_id():
    chat_id = None
    
    # 1. Try to load from existing .env file
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE, "r") as f:
            content = f.read()
            match = re.search(r"TELEGRAM_CHAT_ID\s*=\s*['\"]?(\d+)['\"]?", content)
            if match:
                chat_id = match.group(1)
                print(f"🔹 Loaded Telegram Chat ID from .env: {chat_id}")
                return chat_id

    # 2. If not found, retrieve it automatically by checking Bot updates
    print("\n" + "="*60)
    print("📢 TELEGRAM CONFIGURATION REQUIREMENT")
    print("="*60)
    print("1. Open Telegram on your phone or computer.")
    print("2. Search for your Bot (using your bot's username).")
    print("3. Click 'START' or send any message to it (e.g. 'hello').")
    print("="*60)
    input("👉 AFTER you have sent a message to the bot, press ENTER here to link it...")

    print("\n🔍 Fetching Chat ID from Telegram server...")
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            results = data.get("result", [])
            if not results:
                print("❌ No messages found! Please make sure you sent a message to your bot first, then try running this again.")
                return None
            
            # Get chat id from the last message
            last_message = results[-1]
            chat = last_message.get("message", {}).get("chat", {})
            chat_id = chat.get("id")
            first_name = chat.get("first_name", "User")
            
            if chat_id:
                print(f"✅ Found Chat ID: {chat_id} for user: {first_name}!")
                
                # Append to .env file
                with open(ENV_FILE, "a") as f:
                    f.write(f"\n# Cloudflare Tunnel Telegram Config\nTELEGRAM_CHAT_ID={chat_id}\n")
                print("💾 Saved TELEGRAM_CHAT_ID to .env file!")
                return str(chat_id)
    except Exception as e:
        print(f"❌ Error getting Chat ID: {e}")
        
    return None

def send_telegram_message(chat_id, text):
    print(f"📤 Sending notification to Telegram...")
    encoded_text = urllib.parse.quote(text)
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage?chat_id={chat_id}&text={encoded_text}&parse_mode=Markdown"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            res = json.loads(response.read().decode())
            if res.get("ok"):
                print("✅ Telegram notification sent successfully!")
                return True
    except Exception as e:
        print(f"❌ Failed to send Telegram message: {e}")
    return False

def start_tunnel_and_notify(chat_id):
    if not os.path.exists(CLOUDFLARED_PATH):
        print(f"❌ Could not find cloudflared at {CLOUDFLARED_PATH}")
        print("Please verify the installation path.")
        return

    print(f"\n🚀 Starting Cloudflare Tunnel to http://localhost:{PORT}...")
    
    # Start the tunnel process and redirect stderr to stdout to parse logs
    process = subprocess.Popen(
        [CLOUDFLARED_PATH, "tunnel", "--url", f"http://localhost:{PORT}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    tunnel_url = None
    try:
        # Read the logs line-by-line in real-time
        for line in iter(process.stdout.readline, ''):
            # Print the cloudflared output line to the local console
            print(f"[cloudflared] {line.strip()}")
            
            # Look for the trycloudflare.com URL pattern
            if "trycloudflare.com" in line:
                match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
                if match:
                    tunnel_url = match.group(0)
                    print("\n" + "*"*80)
                    print(f"✨ FOUND PUBLIC URL: {tunnel_url}")
                    print("*"*80 + "\n")
                    
                    # Send telegram message!
                    msg = (
                        f"🚀 *Your Local Face System is Online!*\n\n"
                        f"🔗 *Tunnel URL:* {tunnel_url}\n\n"
                        f"📱 Click the link to access it from any network without VPN!"
                    )
                    send_telegram_message(chat_id, msg)
                    print("\n💡 Tunnel is active and running in the background. Keep this window open.")
                    print("Press Ctrl+C to close the tunnel.\n")
                    
            # Keep reading the output so the buffer doesn't fill up
    except KeyboardInterrupt:
        print("\nStopping Cloudflare Tunnel...")
    finally:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
        print("🔒 Tunnel closed.")

if __name__ == "__main__":
    chat_id = load_or_get_chat_id()
    if chat_id:
        start_tunnel_and_notify(chat_id)
    else:
        print("❌ Could not proceed without a Telegram Chat ID.")
