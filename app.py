from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

# Khởi tạo Flask app
app = Flask(__name__)

# Load mô hình đã huấn luyện
model = load_model('ecg_cnn_model.h5')

# Định nghĩa route cho API để xử lý các yêu cầu POST
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Lấy dữ liệu ảnh từ request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']

        # Kiểm tra xem file có hợp lệ hay không
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        # Mở và xử lý ảnh
        img = Image.open(io.BytesIO(file.read()))
        img = img.resize((64, 64))  # Resize về kích thước phù hợp với mô hình
        img = np.array(img)
        img = img.astype('float32') / 255.0  # Chuẩn hóa ảnh
        img = np.expand_dims(img, axis=0)  # Thêm batch dimension
        
        # Sử dụng mô hình để dự đoán
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions, axis=1)
        
        # Trả về phản hồi dưới dạng JSON
        return jsonify({
            'prediction': int(predicted_class[0]),  # Lấy nhãn dự đoán
            'confidence': float(np.max(predictions))  # Lấy xác suất lớn nhất
        })
    except Exception as e:
        return jsonify({'error': str(e)})

# Chạy app
if __name__ == '__main__':
    app.run(debug=True)
