# Qwen Coder और Pytorch के लिए एक मजबूत बेस इमेज का उपयोग करें
# (Use a robust base image for Qwen Coder and Pytorch)
FROM nvcr.io/nvidia/pytorch:23.10-py3

# AI कोर के लिए जरूरी एनवायरनमेंट वैरिएबल्स सेट करें
ENV PYTHONUNBUFFERED 1
ENV AUMCORE_AI_VERSION 1.0.0-phase2

# डिप्लॉयमेंट फोल्डर को कॉपी करें और निर्भरताएँ इंस्टॉल करें
COPY deployment/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# प्रोजेक्ट फोल्डर सेटअप करें
WORKDIR /app
COPY . /app

# मुख्य स्टार्टअप स्क्रिप्ट चलाएँ
CMD ["/app/deployment/deploy.sh"]