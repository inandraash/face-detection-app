from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
import base64
from flask_cors import CORS
import face_recognition

app = Flask(__name__)

# Enable CORS agar bisa diakses dari domain Laravel/Client lain
CORS(app) 

# Batasi ukuran upload (16MB) untuk mencegah memory overflow
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  

# Load pre-trained face cascade classifier (untuk deteksi cepat di preview)
# Pastikan file xml haarcascade tersedia via cv2.data
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def decode_base64_image(image_data):
    """
    Decode base64 image ke format OpenCV.
    Fungsi ini otomatis menangani format JPG/PNG tanpa peduli ekstensi file,
    sehingga menghindari error libpng.
    """
    try:
        # 1. Bersihkan header data URI (contoh: "data:image/jpeg;base64,...")
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # 2. Decode base64 string ke bytes
        image_bytes = base64.b64decode(image_data)
        
        # 3. Convert bytes ke numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # 4. Decode image menggunakan OpenCV
        # cv2.imdecode pintar mendeteksi header file asli (JPG/PNG) dari bytes
        image_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return image_cv2
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def detect_faces_in_image(image_cv2):
    """
    Deteksi wajah cepat menggunakan Haar Cascade (OpenCV).
    Digunakan hanya untuk menghitung jumlah wajah, bukan untuk mencocokkan identitas.
    """
    gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=4,
        minSize=(30, 30)
    )
    return faces

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint untuk memastikan service jalan"""
    return jsonify({
        'success': True,
        'status': 'running',
        'message': 'Flask Face Recognition API is running correctly'
    })

# ---------------------------------------------------------
# API 1: Deteksi Wajah Cepat (Untuk Preview Kamera)
# ---------------------------------------------------------
@app.route('/api/detect-face-frame', methods=['POST'])
def api_detect_face_frame():
    """
    Menerima frame video, mengembalikan jumlah wajah yang terdeteksi.
    Ringan dan cepat.
    """
    try:
        data = request.json
        if not data or 'frame' not in data:
            return jsonify({'success': False, 'error': 'Tidak ada frame dikirim'}), 400
        
        image_cv2 = decode_base64_image(data['frame'])
        
        if image_cv2 is None:
            return jsonify({'success': False, 'face_count': 0, 'error': 'Gagal decode gambar'})
        
        # Deteksi wajah pakai Haar Cascade
        faces = detect_faces_in_image(image_cv2)
        
        return jsonify({
            'success': True,
            'face_detected': len(faces) > 0,
            'face_count': len(faces),
            'message': f'{len(faces)} wajah terdeteksi' if len(faces) > 0 else 'Tidak ada wajah'
        })
    
    except Exception as e:
        print(f"Error di detect-face-frame: {e}")
        return jsonify({'success': False, 'face_count': 0, 'error': str(e)}), 500

# ---------------------------------------------------------
# API 2: Validasi / Pencocokan Wajah (Presensi)
# ---------------------------------------------------------
@app.route('/api/validate-face', methods=['POST'])
def api_validate_face():
    """
    Membandingkan foto input (kamera) dengan foto referensi (database).
    Menggunakan library face_recognition (Dlib) yang akurat.
    """
    try:
        data = request.json
        if not data or 'photo' not in data or 'reference_photo' not in data:
            return jsonify({'success': False, 'error': 'Data photo dan reference_photo diperlukan'}), 400
        
        # 1. Decode Gambar
        photo_cv2 = decode_base64_image(data['photo'])
        reference_cv2 = decode_base64_image(data['reference_photo'])
        
        if photo_cv2 is None or reference_cv2 is None:
            return jsonify({'success': False, 'error': 'Gagal memproses gambar (File rusak atau format tidak didukung)'}), 400
        
        # 2. Convert BGR (OpenCV) ke RGB (face_recognition butuh RGB)
        photo_rgb = cv2.cvtColor(photo_cv2, cv2.COLOR_BGR2RGB)
        reference_rgb = cv2.cvtColor(reference_cv2, cv2.COLOR_BGR2RGB)
        
        # 3. Cari lokasi wajah di foto input
        photo_locations = face_recognition.face_locations(photo_rgb)
        if len(photo_locations) == 0:
            return jsonify({
                'success': False, 
                'match': False, 
                'error': 'Wajah tidak ditemukan di kamera. Pastikan pencahayaan cukup.',
                'face_count': 0
            })
        
        # 4. Cari lokasi wajah di foto referensi
        ref_locations = face_recognition.face_locations(reference_rgb)
        if len(ref_locations) == 0:
            return jsonify({
                'success': False, 
                'match': False, 
                'error': 'Foto profil database tidak valid (wajah tidak terbaca). Silakan update foto profil.',
                'reference_face_count': 0
            })
            
        # 5. Encoding (Ekstrak fitur biometrik wajah)
        # Ambil wajah pertama saja [0]
        photo_encoding = face_recognition.face_encodings(photo_rgb, photo_locations)[0]
        reference_encoding = face_recognition.face_encodings(reference_rgb, ref_locations)[0]
        
        # 6. Hitung Jarak (Euclidean Distance)
        # Semakin kecil distance = semakin mirip
        face_distance = face_recognition.face_distance([reference_encoding], photo_encoding)[0]
        
        # --- SETTING SENSITIVITAS ---
        # Default library adalah 0.6 (agak longgar)
        # Kita set ke 0.50 agar lebih ketat untuk presensi
        # Jika masih sering salah orang, turunkan ke 0.45
        THRESHOLD = 0.50
        
        is_match = face_distance < THRESHOLD
        
        # Hitung persentase kemiripan (hanya estimasi visual)
        similarity = max(0, (1 - face_distance) * 100)
        
        print(f"[LOG] Distance: {face_distance:.4f} | Threshold: {THRESHOLD} | Match: {is_match}")
        
        return jsonify({
            'success': True,
            'match': bool(is_match),
            'similarity': round(similarity, 2),
            'distance': round(float(face_distance), 4),
            'threshold': THRESHOLD,
            'message': 'Verifikasi Berhasil' if is_match else 'Wajah tidak cocok dengan data pegawai',
            'face_count': len(photo_locations)
        })
    
    except Exception as e:
        print(f"Error validation: {str(e)}")
        return jsonify({'success': False, 'error': f"Server Error: {str(e)}"}), 500

if __name__ == '__main__':
    # Ambil PORT dari environment variable (Wajib untuk Railway)
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)