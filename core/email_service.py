import os
import smtplib
from email.message import EmailMessage

from dotenv import load_dotenv


def send_email(sender_email: str, app_password: str, receiver_email: str, file_path: str) -> bool:
    """
    Send email with CSV attachment using Gmail SMTP.
    Returns True on success, False on failure.
    """
    msg = EmailMessage()
    msg["Subject"] = "Daily Attendance Summary"
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg.set_content("Please find attached the daily attendance summary.")

    debug = os.getenv("EMAIL_DEBUG") == "1"
    try:
        with open(file_path, "rb") as f:
            file_data = f.read()
            file_name = os.path.basename(file_path)
        msg.add_attachment(file_data, maintype="text", subtype="csv", filename=file_name)
    except Exception as exc:
        if debug:
            print(f"Attachment error: {exc}")
        return False

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender_email, app_password)
            smtp.send_message(msg)
        return True
    except Exception as exc:
        if debug:
            print(f"SMTP error: {exc}")
        return False


def load_email_env():
    load_dotenv()
    sender = os.getenv("EMAIL_SENDER")
    app_password = os.getenv("EMAIL_APP_PASSWORD")
    receiver = os.getenv("EMAIL_RECEIVER")
    return sender, app_password, receiver
