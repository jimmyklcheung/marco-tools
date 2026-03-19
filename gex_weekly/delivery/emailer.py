"""
SMTP email sender with PDF attachment.
Never raises exceptions — scheduler must not die on email failure.
"""
import logging
import os
import smtplib
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List

logger = logging.getLogger(__name__)


def send_report(
    report_path: str,
    subject: str,
    html_body: str,
    config: dict,
) -> bool:
    """
    Send the weekly GEX report via SMTP with PDF attachment.

    Args:
        report_path: Filesystem path to the PDF file.
        subject:     Email subject line.
        html_body:   Rendered HTML email body string.
        config:      Full config dict (reads from config["email"]).

    Returns:
        True on success, False on any failure.
    """
    email_cfg = config.get("email", {})

    if not email_cfg.get("enabled", False):
        logger.info("Email delivery disabled in config.")
        return False

    smtp_host = email_cfg.get("smtp_host", "smtp.gmail.com")
    smtp_port = int(email_cfg.get("smtp_port", 587))
    smtp_user = email_cfg.get("smtp_user", "")
    smtp_password = email_cfg.get("smtp_password", "")
    from_name = email_cfg.get("from_name", "Macro GEX Report")
    from_address = email_cfg.get("from_address", smtp_user)
    recipients: List[str] = email_cfg.get("recipients", [])

    if not recipients:
        logger.warning("No email recipients configured.")
        return False

    try:
        # Build message
        msg = MIMEMultipart("mixed")
        msg["Subject"] = subject
        msg["From"] = f"{from_name} <{from_address}>"
        msg["To"] = ", ".join(recipients)

        # HTML body part
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        # PDF attachment
        if report_path and os.path.exists(report_path) and report_path.endswith(".pdf"):
            with open(report_path, "rb") as f:
                pdf_data = f.read()
            part = MIMEBase("application", "pdf")
            part.set_payload(pdf_data)
            encoders.encode_base64(part)
            filename = os.path.basename(report_path)
            part.add_header("Content-Disposition", f'attachment; filename="{filename}"')
            msg.attach(part)
        else:
            logger.warning(f"PDF not found or not a PDF: {report_path} — sending without attachment")

        # Send
        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            if smtp_user and smtp_password:
                server.login(smtp_user, smtp_password)
            server.sendmail(from_address, recipients, msg.as_string())

        logger.info(f"Email sent to {len(recipients)} recipients: {', '.join(recipients)}")
        return True

    except Exception as e:
        logger.error(f"Email failed: {e}")
        return False
