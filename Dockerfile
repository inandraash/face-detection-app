# Strategi "Pure Debian": Menginstal semua library via APT Repository
# Ini adalah metode paling stabil karena semua library (dlib, face_recognition)
# sudah berupa binary yang teruji 100% kompatibel dengan Debian 12.
FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive

# Update & Install Paket
# - python3-face-recognition: include dlib & models (Binary)
# - python3-opencv: Binary opencv
# - python3-flask-cors: Binary extension
# - gunicorn: Binary server
RUN apt-get update && apt-get install -y \
    python3 \
    python3-face-recognition \
    python3-flask \
    python3-flask-cors \
    python3-opencv \
    gunicorn \
    && rm -rf /var/lib/apt/lists/*

# Setup User Hugging Face
RUN useradd -m -u 1000 user
WORKDIR /app

# Copy File
# Mengcopy titik (.) artinya mengcopy semua file yang diupload
# ke dalam folder /app di container
COPY --chown=user . /app

# Bersiap jalan
USER user
EXPOSE 7860

# Jalankan Gunicorn
# Kita set timeout panjang karena dlib butuh waktu load model di awal
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app", "--timeout", "120"]
