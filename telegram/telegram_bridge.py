"""
Telegram ↔ Claude Code Bridge
Sends Telegram messages to Claude CLI and returns the response.
"""

import subprocess
import os
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8512768565:AAE1sMTam-sLaFciMeM8Hqm_uP8Y8kxiiqY")
CLAUDE_PATH = r"C:\Users\Administrator\.local\bin\claude.exe"
WORK_DIR = r"C:\Users\Administrator\Documents\GitHub\vbt_vc"

# Only allow your own Telegram user ID (set via env var for security)
ALLOWED_USER_ID = os.environ.get("TELEGRAM_USER_ID")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    await update.message.reply_text(
        f"Claude Code Bridge aktiv.\n"
        f"Deine User-ID: {user_id}\n\n"
        f"Sende mir eine Nachricht und ich leite sie an Claude weiter.\n"
        f"Tipp: Setze TELEGRAM_USER_ID={user_id} um den Bot auf dich zu beschränken."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)

    # Security: only allow authorized user
    if ALLOWED_USER_ID and user_id != ALLOWED_USER_ID:
        await update.message.reply_text("Nicht autorisiert.")
        return

    prompt = update.message.text
    logger.info(f"Received from {user_id}: {prompt[:80]}...")

    await update.message.reply_text("⏳ Claude denkt nach...")

    try:
        result = subprocess.run(
            [CLAUDE_PATH, "-p", prompt, "--output-format", "text"],
            capture_output=True,
            text=True,
            cwd=WORK_DIR,
            timeout=300,
            encoding="utf-8",
        )

        response = result.stdout.strip() or result.stderr.strip() or "Keine Antwort von Claude."

        # Telegram max message length is 4096
        if len(response) > 4000:
            for i in range(0, len(response), 4000):
                await update.message.reply_text(response[i:i+4000])
        else:
            await update.message.reply_text(response)

    except subprocess.TimeoutExpired:
        await update.message.reply_text("⏰ Timeout — Claude hat zu lange gebraucht.")
    except Exception as e:
        await update.message.reply_text(f"Fehler: {e}")


def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot gestartet. Warte auf Nachrichten...")
    app.run_polling()


if __name__ == "__main__":
    main()
