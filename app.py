from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
import base64
from flask_cors import CORS
import face_recognition

app = Flask(__name__)
CORS(app)  # Enable CORS for Laravel integration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load pre-trained face cascade classifier untuk deteksi cepat
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def decode_base64_image(image_data):
    """Decode base64 image ke format OpenCV"""
    if image_data.startswith('data:image'):
        image_data = image_data.split(',')[1]
    
    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image_cv2

def detect_faces_in_image(image_cv2):
    """Deteksi wajah dalam gambar menggunakan OpenCV (cepat untuk preview)"""
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
    """Health check endpoint"""
    return jsonify({
        'success': True,
        'status': 'running',
        'message': 'Flask Face Recognition API is running'
    })

# API Endpoint untuk deteksi wajah dari video stream (base64)
@app.route('/api/detect-face-frame', methods=['POST'])
def api_detect_face_frame():
    """
    API endpoint untuk deteksi wajah dari frame video
    Menerima frame video sebagai base64
    Returns: JSON dengan hasil deteksi (face_count only untuk performance)
    """
    try:
        data = request.json
        if not data or 'frame' not in data:
            return jsonify({'success': False, 'error': 'Tidak ada frame'}), 400
        
        image_cv2 = decode_base64_image(data['frame'])
        
        if image_cv2 is None:
            return jsonify({'success': False, 'face_count': 0})
        
        # Deteksi wajah
        faces = detect_faces_in_image(image_cv2)
        
        return jsonify({
            'success': True,
            'face_detected': len(faces) > 0,
            'face_count': len(faces),
            'message': f'{len(faces)} wajah terdeteksi' if len(faces) > 0 else 'Tidak ada wajah'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'face_count': 0, 'error': str(e)})

@app.route('/api/validate-face', methods=['POST'])
def api_validate_face():
    """
    API endpoint untuk validasi wajah dengan foto referensi
    Menerima:
    - photo: foto yang akan divalidasi (base64)
    - reference_photo: foto referensi pegawai (base64) 
    Returns: JSON dengan hasil validasi (match/tidak match) dan similarity score
    """
    try:
        data = request.json
        if not data or 'photo' not in data or 'reference_photo' not in data:
            return jsonify({
                'success': False, 
                'error': 'Photo dan reference_photo harus dikirim'
            }), 400
        
        # Decode kedua gambar
        photo_cv2 = decode_base64_image(data['photo'])
        reference_cv2 = decode_base64_image(data['reference_photo'])
        
        if photo_cv2 is None or reference_cv2 is None:
            return jsonify({
                'success': False, 
                'error': 'Gagal membaca gambar'
            }), 400
        
        # Convert BGR (OpenCV) to RGB (face_recognition)
        photo_rgb = cv2.cvtColor(photo_cv2, cv2.COLOR_BGR2RGB)
        reference_rgb = cv2.cvtColor(reference_cv2, cv2.COLOR_BGR2RGB)
        
        # Deteksi dan encode wajah di foto yang akan divalidasi
        photo_face_locations = face_recognition.face_locations(photo_rgb)
        if len(photo_face_locations) == 0:
            return jsonify({
                'success': False,
                'match': False,
                'error': 'Tidak ada wajah terdeteksi di foto',
                'face_count': 0
            })
        
        if len(photo_face_locations) > 1:
            return jsonify({
                'success': False,
                'match': False,
                'error': 'Terdeteksi lebih dari 1 wajah. Pastikan hanya ada 1 wajah',
                'face_count': len(photo_face_locations)
            })
        
        photo_encodings = face_recognition.face_encodings(photo_rgb, photo_face_locations)
        if len(photo_encodings) == 0:
            return jsonify({
                'success': False,
                'match': False,
                'error': 'Gagal mengekstrak fitur wajah dari foto'
            })
        
        photo_encoding = photo_encodings[0]
        
        # Deteksi dan encode wajah di foto referensi
        reference_face_locations = face_recognition.face_locations(reference_rgb)
        if len(reference_face_locations) == 0:
            return jsonify({
                'success': False,
                'match': False,
                'error': 'Tidak ada wajah terdeteksi di foto referensi',
                'reference_face_count': 0
            })
        
        reference_encodings = face_recognition.face_encodings(reference_rgb, reference_face_locations)
        if len(reference_encodings) == 0:
            return jsonify({
                'success': False,
                'match': False,
                'error': 'Gagal mengekstrak fitur wajah dari foto referensi'
            })
        
        reference_encoding = reference_encodings[0]
        
        # Bandingkan wajah
        # face_distance: semakin kecil semakin mirip (0 = identik, >0.6 = beda orang)
        face_distance = face_recognition.face_distance([reference_encoding], photo_encoding)[0]
        
        # Threshold untuk menentukan match (0.6 adalah default, bisa disesuaikan)
        # Untuk sistem presensi, gunakan threshold yang lebih ketat (0.5)
        threshold = 0.6
        is_match = face_distance < threshold
        
        # Konversi distance ke similarity score (0-100%)
        similarity = max(0, (1 - face_distance) * 100)
        
        return jsonify({
            'success': True,
            'match': bool(is_match),
            'similarity': round(similarity, 2),
            'distance': round(float(face_distance), 4),
            'threshold': threshold,
            'message': 'Wajah cocok' if is_match else 'Wajah tidak cocok',
            'face_count': len(photo_face_locations),
            'reference_face_count': len(reference_face_locations)
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    # Railway menyediakan port lewat environment variable 'PORT'
    port = int(os.environ.get('PORT', os.environ.get('FLASK_SERVER_PORT', 5000)))
    app.run(debug=False, host='0.0.0.0', port=port, use_reloader=False)
