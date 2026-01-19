# Gunakan Image Python Official (Stabil & Kompatibel)
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1

# Install System Dependencies
# - cmake & build-essential: Wajib untuk compile dlib (jika tidak ada wheel)
# - libgl1 & libglib2.0: Dependencies wajib untuk OpenCV
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

# Upgrade pip dan install requirements
# Railway biasanya punya cache untuk pip, jadi build kedua lebih cepat
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Perintah menjalankan aplikasi (sesuai struktur src/)
CMD ["sh", "-c", "gunicorn --chdir src --bind 0.0.0.0:${PORT:-5000} app:app --timeout 120"]
