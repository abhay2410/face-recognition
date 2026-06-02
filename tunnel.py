"""
tunnel.py — Cloudflare Tunnel Manager + Telegram Security Bot
=============================================================
Features:
  - Starts cloudflared in a background thread on FastAPI startup.
  - Parses tunnel URL and sends it ONLY to whitelisted Telegram user IDs.
  - Runs an interactive Telegram polling loop — unauthorized users are
    silently ignored; authorized users can type /url to get the live link.
  - Cleanly shuts down on FastAPI shutdown.

Security Model:
  TELEGRAM_ALLOWED_IDS in .env = comma-separated list of authorized chat IDs.
  Only those IDs will ever receive a message from this bot.
"""

import re
import subprocess
import threading
import logging
import urllib.request
import urllib.parse
import json
import os
import config

log = logging.getLogger("tunnel")

# ── Global State ──────────────────────────────────────────────────────────────
_tunnel_process  = None
_monitor_thread  = None
_bot_thread      = None
_stop_event      = threading.Event()
_current_url     = None   # Stores the latest live tunnel URL


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_allowed_ids() -> set:
    """Return set of int chat IDs that are allowed to receive messages."""
    raw = os.getenv("TELEGRAM_ALLOWED_IDS", "")
    ids = set()
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            ids.add(int(part))
    # Always include the primary TELEGRAM_CHAT_ID as allowed
    if config.TELEGRAM_CHAT_ID and str(config.TELEGRAM_CHAT_ID).isdigit():
        ids.add(int(config.TELEGRAM_CHAT_ID))
    return ids


def _send_message(chat_id: int, text: str) -> bool:
    """Send a Telegram message to a specific chat_id."""
    if not config.TELEGRAM_BOT_TOKEN:
        return False
    encoded = urllib.parse.quote(text)
    url = (
        f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}"
        f"/sendMessage?chat_id={chat_id}&text={encoded}&parse_mode=Markdown"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            res = json.loads(resp.read().decode())
            return res.get("ok", False)
    except Exception as e:
        log.error("[Tunnel] Failed to send Telegram message: %s", e)
    return False


def _broadcast_to_allowed(text: str):
    """Send a message to ALL whitelisted user IDs."""
    for uid in _get_allowed_ids():
        ok = _send_message(uid, text)
        if ok:
            log.info("[Tunnel] Notified user %d via Telegram.", uid)
        else:
            log.warning("[Tunnel] Failed to notify user %d.", uid)


# ── Tunnel Monitor ────────────────────────────────────────────────────────────

def _monitor_tunnel():
    """Background thread: parse cloudflared output for the public URL."""
    global _current_url

    log.info("[Tunnel] Monitor thread started.")
    try:
        for line in iter(_tunnel_process.stdout.readline, ""):
            if _stop_event.is_set():
                break

            line = line.strip()
            if "trycloudflare.com" in line:
                match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
                if match:
                    _current_url = match.group(0)
                    log.info("[Tunnel] ✅ Public URL: %s", _current_url)

                    msg = (
                        f"🚀 *Face Recognition System is Online!*\n\n"
                        f"🔗 *Tunnel URL:* {_current_url}\n\n"
                        f"🔒 This link is private — shared only with authorized users.\n"
                        f"📱 Tap the link to open on any network!"
                    )
                    _broadcast_to_allowed(msg)

            elif "ERR" in line:
                log.debug("[Tunnel] %s", line)

    except Exception as e:
        log.error("[Tunnel] Monitor error: %s", e)


# ── Interactive Telegram Bot (Command Listener) ───────────────────────────────

def _bot_polling_loop():
    """
    Background thread: poll Telegram for commands.
    Responds to authorized users only. Unauthorized senders are silently dropped.
    
    Supported commands:
      /url    — Reply with the current live tunnel URL
      /status — Reply with system status
    """
    offset = 0
    log.info("[TelegramBot] Polling loop started.")

    while not _stop_event.is_set():
        try:
            url = (
                f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}"
                f"/getUpdates?timeout=20&offset={offset}"
            )
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = json.loads(resp.read().decode())

            for update in data.get("result", []):
                offset = update["update_id"] + 1
                msg    = update.get("message", {})
                if not msg:
                    continue

                sender_id   = msg.get("chat", {}).get("id")
                sender_name = msg.get("chat", {}).get("first_name", "Unknown")
                text        = msg.get("text", "").strip().lower()

                # ── Security Gate ─────────────────────────────────────────────
                allowed = _get_allowed_ids()
                if sender_id not in allowed:
                    log.warning(
                        "[TelegramBot] 🚫 Blocked unauthorized access from user %s (ID: %d).",
                        sender_name, sender_id
                    )
                    # Do NOT reply — silently ignore to avoid confirming bot existence
                    continue

                # ── Authorized User Commands ──────────────────────────────────
                log.info("[TelegramBot] Command '%s' from authorized user %s (%d).", text, sender_name, sender_id)

                if "/url" in text or "/link" in text:
                    if _current_url:
                        reply = (
                            f"🔗 *Current Tunnel URL:*\n{_current_url}\n\n"
                            f"✅ System is running. Login with your credentials."
                        )
                    else:
                        reply = "⏳ Tunnel is starting up, please try again in a moment."
                    _send_message(sender_id, reply)

                elif "/status" in text:
                    status = "🟢 Online" if _current_url else "🔴 Offline / Starting"
                    reply = (
                        f"📊 *Face Recognition System Status*\n\n"
                        f"Tunnel: {status}\n"
                        f"URL: {_current_url or 'Not available yet'}"
                    )
                    _send_message(sender_id, reply)

                elif "/help" in text or "/start" in text:
                    reply = (
                        f"👋 Hello *{sender_name}*! You are an authorized user.\n\n"
                        f"Available commands:\n"
                        f"  /url — Get the current tunnel link\n"
                        f"  /status — Check system status\n"
                        f"  /help — Show this message"
                    )
                    _send_message(sender_id, reply)

        except Exception as e:
            if not _stop_event.is_set():
                log.debug("[TelegramBot] Polling error (retrying): %s", e)

    log.info("[TelegramBot] Polling loop stopped.")


# ── Public API ────────────────────────────────────────────────────────────────

def start_background_tunnel():
    """Start Cloudflare Tunnel + Telegram bot. Called from FastAPI lifespan."""
    global _tunnel_process, _monitor_thread, _bot_thread

    if not config.CLOUDFLARE_TUNNEL_ENABLED:
        log.info("[Tunnel] Disabled in config (CLOUDFLARE_TUNNEL_ENABLED=false).")
        return

    if not os.path.exists(config.CLOUDFLARED_PATH):
        log.warning("[Tunnel] cloudflared not found at '%s'. Skipping.", config.CLOUDFLARED_PATH)
        return

    if not config.TELEGRAM_BOT_TOKEN:
        log.warning("[Tunnel] TELEGRAM_BOT_TOKEN not set. Skipping notifications.")
        return

    allowed = _get_allowed_ids()
    if not allowed:
        log.warning("[Tunnel] No authorized Telegram users configured. Set TELEGRAM_CHAT_ID or TELEGRAM_ALLOWED_IDS.")
        return

    log.info("[Tunnel] Authorized Telegram users: %s", allowed)
    _stop_event.clear()

    # Launch cloudflared process
    creationflags = 0
    if os.name == "nt":
        creationflags = 0x08000000  # subprocess.CREATE_NO_WINDOW

    try:
        _tunnel_process = subprocess.Popen(
            [config.CLOUDFLARED_PATH, "tunnel", "--url", f"http://localhost:{config.API_PORT}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=creationflags,
        )
    except Exception as e:
        log.error("[Tunnel] Failed to start cloudflared: %s", e)
        return

    # Thread 1: Monitor tunnel output for URL
    _monitor_thread = threading.Thread(target=_monitor_tunnel, daemon=True, name="TunnelMonitor")
    _monitor_thread.start()

    # Thread 2: Poll Telegram for commands from authorized users
    _bot_thread = threading.Thread(target=_bot_polling_loop, daemon=True, name="TelegramBot")
    _bot_thread.start()

    log.info("[Tunnel] Cloudflare Tunnel + Telegram Bot started successfully.")


def stop_background_tunnel():
    """Cleanly stop tunnel and bot threads. Called from FastAPI lifespan."""
    global _tunnel_process

    log.info("[Tunnel] Shutting down...")
    _stop_event.set()

    if _tunnel_process:
        _tunnel_process.terminate()
        try:
            _tunnel_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            _tunnel_process.kill()
        _tunnel_process = None

    log.info("[Tunnel] Shutdown complete.")
