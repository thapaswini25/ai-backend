from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from detection import detect_fake_news, detect_ai_media
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/')
def index():
    return "Backend is running"
@app.route('/detect-news', methods=['POST'])
def detect_news():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    result = detect_fake_news(text)
    return jsonify({'result': result})
@app.route('/detect-media', methods=['POST'])
def detect_media():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    result = detect_ai_media(filepath)
    os.remove(filepath)
    return jsonify({'result': result})
if __name__ == '__main__':
    app.run(debug=True) 
