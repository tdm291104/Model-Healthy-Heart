import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras import regularizers

# Đường dẫn tới folder chứa dataset
base_dir = 'D:\\Learn\\Python\\PBL4\\processed_dataset\\Train\\'
val_dir = 'D:\\Learn\\Python\\PBL4\\processed_dataset\\Val\\'
test_dir = 'D:\\Learn\\Python\\PBL4\\processed_dataset\\Test\\'

# Hàm khởi tạo các ImageDataGenerator
def create_generators(base_dir, val_dir, test_dir):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        base_dir,
        target_size=(64, 64),
        batch_size=32,
        class_mode='sparse'
    )

    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(64, 64),
        batch_size=32,
        class_mode='sparse'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(64, 64),
        batch_size=32,
        class_mode='sparse'
    )

    return train_generator, validation_generator, test_generator

# Hàm xây dựng mô hình đơn giản (model_v1)
def create_model_v1(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Hàm xây dựng mô hình CNN, 3 lớp tích chập, 3 lớp kích hoạt ReLU, 3 lớp max pooling
def create_model_v2(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=regularizers.l2(0.001)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),  # Dropout layer
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Hàm huấn luyện mô hình
def train_model(model, train_generator, validation_generator, epochs=5):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator
    )
    return history

# Hàm đánh giá mô hình trên tập kiểm tra
def evaluate_model(model, test_generator):
    test_loss, test_acc = model.evaluate(test_generator, verbose=2)
    print('\nTest accuracy:', test_acc)
    return test_loss, test_acc

# Hàm vẽ đồ thị accuracy và loss
def plot_metrics(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Hàm vẽ confusion matrix
def plot_confusion_matrix(model, generator, class_names, dataset_type='Test'):
    predictions = model.predict(generator)
    y_true = generator.classes
    y_pred = np.argmax(predictions, axis=1)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.title(f'Confusion Matrix - {dataset_type} Dataset')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


train_generator, validation_generator, test_generator = create_generators(base_dir, val_dir, test_dir)
class_names = list(train_generator.class_indices.keys())
model = create_model_v2((64, 64, 3), len(class_names))

# Huấn luyện mô hình
history = train_model(model, train_generator, validation_generator, epochs=10)

# Vẽ đồ thị
plot_metrics(history)

# Vẽ confusion matrix cho tập huấn luyện (train)
plot_confusion_matrix(model, train_generator, class_names, dataset_type='Train')

# Vẽ confusion matrix cho tập xác thực (validation)
plot_confusion_matrix(model, validation_generator, class_names, dataset_type='Validation')

# Vẽ confusion matrix cho tập kiểm tra (test)
plot_confusion_matrix(model, test_generator, class_names, dataset_type='Test')

# Lưu mô hình nếu cần
model.save('ecg_cnn_model.h5')
