import os
import platform
import smtplib
from email.mime.text import MIMEText

def send_notification(subject, message, to_email=None):
    """Send notification via desktop (Windows) and/or email."""
    # Desktop notification (Windows only)
    if platform.system() == "Windows":
        try:
            from win10toast import ToastNotifier
            toaster = ToastNotifier()
            toaster.show_toast(subject, message, duration=10)
        except Exception:
            pass
    # Email notification (if to_email provided)
    if to_email:
        try:
            msg = MIMEText(message)
            msg["Subject"] = subject
            msg["From"] = "noreply@projectp.local"
            msg["To"] = to_email
            with smtplib.SMTP("localhost") as server:
                server.send_message(msg)
        except Exception:
            pass
