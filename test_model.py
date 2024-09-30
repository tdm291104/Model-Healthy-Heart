from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Tải mô hình đã huấn luyện
model = load_model('D:\\Learn\\Python\\PBL4\\ecg_cnn_model.keras')
test_dir = 'D:\\Learn\\Python\\PBL4\\processed_dataset\\Test\\'

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='sparse'
)

# Dự đoán nhãn của tập test
predictions = model.predict(test_generator)
predicted_labels = np.argmax(predictions, axis=1)

# Lấy nhãn thực tế
true_labels = test_generator.classes

# Tạo confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Vẽ confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Dự đoán với ảnh mới
# img_path = 'D:\\Learn\\Python\\PBL4\\R.png'
# img = image.load_img(img_path, target_size=(64, 64))
# img = image.img_to_array(img)
# img = np.expand_dims(img, axis=0)

# Dự đoán nhãn
# predictions = model.predict(img)
# predicted_label = np.argmax(predictions[0])

# print(f"Predicted Label: {predicted_label}")
