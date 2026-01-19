# Strategi "Pure Debian" (Revised for Railway):
# Menggunakan python3-dlib dari APT untuk menghindari build dlib yang lama/timeout.
FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive

# Update & Install Paket
# - python3-dlib: Versi binary APT (KUNCI UTAMA: Biar tidak perlu compile!)
# - python3-opencv: Official binary
# - python3-pip: Untuk install face_recognition (wrapper)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dlib \
    python3-opencv \
    python3-flask \
    python3-flask-cors \
    gunicorn \
    && rm -rf /var/lib/apt/lists/*

# Fix: Debian 12 (Bookworm) membatasi pip via "EXTERNALLY-MANAGED", kita hapus proteksinya
RUN rm -rf /usr/lib/python3.*/EXTERNALLY-MANAGED

# Install face-recognition (wrapper ringan, akan pakai dlib dari apt)
RUN pip3 install face-recognition

# Setup User
RUN useradd -m -u 1000 user
WORKDIR /app

# Copy File
COPY --chown=user . /app

# Switch User
USER user

# CMD Support Dynamic Port (Railway)
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} app:app --timeout 120"]
