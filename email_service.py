import asyncio
import smtplib
from email.message import EmailMessage
from typing import List

EMAIL_QUEUE = asyncio.Queue()

async def send_email(recipient: str, pdf_bytes: bytes, filename: str):
    # Build email
    msg = EmailMessage()
    msg["Subject"] = "Your SSM Report"
    msg["From"] = "no-reply@yourdomain.com"
    msg["To"] = recipient
    msg.set_content("Please find your report attached.")
    msg.add_attachment(pdf_bytes, maintype="application", subtype="pdf", filename=filename)

    try:
        # Example: SMTP send (replace with your real server / async email lib)
        with smtplib.SMTP("localhost") as smtp:
            smtp.send_message(msg)
    except Exception as e:
        print(f"[EMAIL ERROR] Failed to send to {recipient}: {e}")
        # optionally re-queue
        await EMAIL_QUEUE.put((recipient, pdf_bytes, filename))

async def email_worker():
    while True:
        recipient, pdf_bytes, filename = await EMAIL_QUEUE.get()
        await send_email(recipient, pdf_bytes, filename)
        EMAIL_QUEUE.task_done()

def queue_email(recipient: str, pdf_bytes: bytes, filename: str):
    EMAIL_QUEUE.put_nowait((recipient, pdf_bytes, filename))
