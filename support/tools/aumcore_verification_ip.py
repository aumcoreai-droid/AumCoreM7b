import random
import smtplib
from email.mime.text import MIMEText
import requests

# ---------------------------
# Brevo SMTP Configuration
# ---------------------------
SMTP_SERVER = "smtp-relay.brevo.com"
SMTP_PORT = 587
SMTP_LOGIN = "9bd240001@smtp-brevo.com"
SMTP_PASSWORD = "QtRmZL4McpYUsOw9"

# ---------------------------
# Step 1: Send 8-digit verification code
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
# Step 2: Check user's public IP
# ---------------------------
def get_user_ip():
    try:
        response = requests.get('https://api.ipify.org?format=json')
        return response.json()['ip']
    except Exception as e:
        return f"Error fetching IP: {e}"

# ---------------------------
# Step 3: User verification flow
# ---------------------------
def user_verification_flow():
    user_email = input("Enter your email: ")
    code_sent = send_verification_code(user_email)
    if not code_sent:
        print("Failed to send verification code. Exiting.")
        return

    user_code = input("Enter the 8-digit code you received: ")
    if user_code == code_sent:
        print("✅ Verification successful!")
        ip = get_user_ip()
        print("🌐 Your Public IP:", ip)
    else:
        print("❌ Verification failed. Code does not match.")

# ---------------------------
# Main execution
# ---------------------------
if __name__ == "__main__":
    user_verification_flow()
