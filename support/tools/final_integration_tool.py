import random
import smtplib
from email.mime.text import MIMEText
import requests
import platform
import subprocess
import os

# ---------------------------
# Brevo SMTP Configuration
# ---------------------------
SMTP_SERVER = "smtp-relay.brevo.com"
SMTP_PORT = 587
SMTP_LOGIN = "9bd240001@smtp-brevo.com"
SMTP_PASSWORD = "QtRmZL4McpYUsOw9"

# ---------------------------
# Send 8-digit verification code
# ---------------------------
def send_verification_code(user_email):
    try:
        code = ''.join([str(random.randint(0, 9)) for _ in range(8)])
        msg = MIMEText(f"Your AumCore AI verification code is: {code}")
        msg['Subject'] = "AumCore AI Verification Code"
        msg['From'] = SMTP_LOGIN
        msg['To'] = user_email

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_LOGIN, SMTP_PASSWORD)
            server.sendmail(SMTP_LOGIN, [user_email], msg.as_string())

        print(f"✅ Verification code sent to {user_email}")
        print(f"🟢 Generated 8-Digit Code (console view for testing): {code}")
        return code
    except Exception as e:
        print("❌ Error sending verification code:", e)
        return None

# ---------------------------
# Get user's public IP
# ---------------------------
def get_user_ip():
    try:
        response = requests.get('https://api.ipify.org?format=json')
        return response.json()['ip']
    except Exception as e:
        return f"Error fetching IP: {e}"

# ---------------------------
# Get system info + BIOS
# ---------------------------
def get_system_info():
    info = {
        'system': platform.system(),
        'node': platform.node(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
        'processor': platform.processor()
    }

    try:
        if os.geteuid() == 0:  # Root permission required
            bios_info = subprocess.check_output('dmidecode -t bios', shell=True, stderr=subprocess.DEVNULL).decode()
            info['bios'] = bios_info
        else:
            info['bios'] = "BIOS info not accessible: Requires root permissions on physical machine."
    except Exception as e:
        info['bios'] = f"BIOS info not available: {e}"

    return info

# ---------------------------
# Full User Verification Flow
# ---------------------------
def user_verification_flow():
    # Send verification code
    user_email = input("Enter your email: ")
    code_sent = send_verification_code(user_email)
    if not code_sent:
        print("Failed to send verification code. Exiting.")
        return

    # User enters received code
    user_code = input("Enter the 8-digit code you received: ")
    if user_code != code_sent:
        print("❌ Verification failed. Code does not match.")
        return
    print("✅ Verification successful!")

    # Get user's public IP
    ip = get_user_ip()
    print("🌐 Your Public IP:", ip)

    # Get system info
    system_info = get_system_info()
    print("💻 System Configuration:")
    for key, value in system_info.items():
        print(f"{key}: {value}")

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == "__main__":
    user_verification_flow()
