import asyncio
import smtplib
from email.message import EmailMessage

# --- Queue for async email sending ---
EMAIL_QUEUE = asyncio.Queue()

# SMTP config
SMTP_HOST = "smtp.yourdomain.com"
SMTP_PORT = 587
SMTP_USER = "noreply@yourdomain.com"
SMTP_PASS = "YOUR_SMTP_PASSWORD"  # or os.environ.get("SMTP_PASSWORD")

async def send_email(recipient: str, pdf_bytes: bytes, filename: str):
    msg = EmailMessage()
    msg["Subject"] = "Your SSM Report"
    msg["From"] = SMTP_USER
    msg["To"] = recipient
    msg.set_content("Please find your report attached.")
    msg.add_attachment(pdf_bytes, maintype="application", subtype="pdf", filename=filename)

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as smtp:
            smtp.starttls()
            smtp.login(SMTP_USER, SMTP_PASS)
            smtp.send_message(msg)
    except Exception as e:
        print(f"[EMAIL ERROR] Failed to send to {recipient}: {e}")
        # Re-queue failed email
        await EMAIL_QUEUE.put((recipient, pdf_bytes, filename))

# --- Background worker that continuously sends queued emails ---
async def email_worker():
    while True:
        recipient, pdf_bytes, filename = await EMAIL_QUEUE.get()
        await send_email(recipient, pdf_bytes, filename)
        EMAIL_QUEUE.task_done()

# --- Helper to queue a new email ---
def queue_email(recipient: str, pdf_bytes: bytes, filename: str):
    EMAIL_QUEUE.put_nowait((recipient, pdf_bytes, filename))
