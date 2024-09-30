import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
import os

# Đọc file ECG
record_name = '115'
record_path = 'D:\\Learn\\Python\\PBL4\\dataset\\'
signals, fields = wfdb.rdsamp(record_path + record_name)
ecg_signal = signals[:, 0]

# Đọc nhãn từ file annotation ('.atr')
annotation = wfdb.rdann(record_path + record_name, 'atr')  # Đọc file annotation
annotation_sample = annotation.sample  # Các vị trí mẫu có nhãn
annotation_label = annotation.symbol  # Các nhãn tương ứng

# Hàm tạo bộ lọc thông dải (band-pass filter)
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Hàm áp dụng bộ lọc thông dải
def bandpass_filter(signal, lowcut=0.5, highcut=40, fs=360, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, signal)
    return y

# Áp dụng bộ lọc thông dải để loại bỏ nhiễu
fs = 360  # Tần số lấy mẫu (Sample rate)
filtered_ecg_signal = bandpass_filter(ecg_signal, lowcut=0.5, highcut=40, fs=fs)

# Tìm đỉnh dương (positive QRS peaks) trong tín hiệu đã lọc
positive_peaks, _ = find_peaks(filtered_ecg_signal, height=None, distance=30)

# Tìm đỉnh âm (negative QRS peaks) bằng cách đảo tín hiệu đã lọc
negative_peaks, _ = find_peaks(-filtered_ecg_signal, height=None, distance=30)

# Gộp đỉnh dương và đỉnh âm
all_peaks = np.sort(np.concatenate((positive_peaks, negative_peaks)))

# Vẽ tín hiệu ECG đã lọc và các đỉnh QRS (dương và âm)
plt.figure(figsize=(12, 6))
plt.plot(filtered_ecg_signal, label='Filtered ECG Signal')
plt.plot(positive_peaks, filtered_ecg_signal[positive_peaks], "x", label='Positive QRS Peaks')
plt.plot(negative_peaks, filtered_ecg_signal[negative_peaks], "o", label='Negative QRS Peaks')
plt.title('Filtered ECG Signal with Detected Positive and Negative QRS Peaks')
plt.xlabel('Sample Number')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.show()

# In ra vị trí của các đỉnh
print("Detected Positive QRS Peaks at Samples:", positive_peaks)
print("Detected Negative QRS Peaks at Samples:", negative_peaks)

# Hàm cắt đoạn tín hiệu quanh đỉnh QRS
def extract_qrs_segments(signal, peaks, window_size=300):  # 300 mẫu = ~0.83 giây (trước và sau đỉnh)
    segments = []
    for peak in peaks:
        start = max(0, peak - window_size // 2)
        end = min(len(signal), peak + window_size // 2)
        segment = signal[start:end]
        if len(segment) == window_size:
            segments.append(segment)
    return np.array(segments)

# Cắt tín hiệu quanh các đỉnh QRS
qrs_segments = extract_qrs_segments(filtered_ecg_signal, all_peaks)

# Lưu các đoạn đã cắt vào từng nhãn dán
labeled_segments = {label: [] for label in set(annotation_label)}

# Phân loại nhãn cho từng đoạn QRS
def get_label_for_peak(peak, annotation_sample, annotation_label):
    # Tìm nhãn gần nhất cho đỉnh (peak) dựa trên vị trí mẫu
    closest_idx = np.argmin(np.abs(annotation_sample - peak))
    return annotation_label[closest_idx]

# Phân loại từng đoạn QRS
for i, segment in enumerate(qrs_segments):
    label = get_label_for_peak(all_peaks[i], annotation_sample, annotation_label)
    labeled_segments[label].append(segment)

# In ra số lượng đoạn đã cắt cho mỗi nhãn
for label, segments in labeled_segments.items():
    print(f"Label '{label}': {len(segments)} segments")

# Hàm vẽ đoạn tín hiệu QRS lên màn hình
def plot_qrs_segment(segment, index, label):
    # Vẽ đoạn tín hiệu QRS
    plt.figure(figsize=(4, 2))
    plt.plot(segment)
    plt.title(f"{record_name} - QRS Segment {index} - Label: {label}")
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

# Hàm tạo thư mục nếu chưa tồn tại
def create_label_directory(label):
    folder_path = os.path.join(record_path, label)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

# Hàm lưu ảnh QRS vào thư mục tương ứng với nhãn
def save_qrs_segment_as_image(segment, index, label):
    # Tạo thư mục cho nhãn nếu chưa tồn tại
    folder_path = create_label_directory(label)
    
    # Vẽ đoạn tín hiệu QRS
    plt.figure(figsize=(4, 2))
    plt.plot(segment)
    # plt.title(f"{record_name} - QRS Segment {index} - Label: {label}")
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    # Lưu ảnh vào thư mục tương ứng
    image_filename = os.path.join(folder_path, f"{record_name}_qrs_segment_{index}.png")
    plt.savefig(image_filename)
    plt.close()  # Đóng file hình ảnh sau khi lưu
    print(f"Saved: {image_filename}")

# Tìm nhãn có số lượng đoạn lớn nhất
max_label = max(labeled_segments, key=lambda label: len(labeled_segments[label]))
max_segments = labeled_segments[max_label]

print(f"Label with the most segments: '{max_label}' with {len(max_segments)} segments")
if 'N' in labeled_segments:
    print(f"label = N {len(labeled_segments['N'])}")
else:
    print("Label 'N' does not exist in the data.")
# Vẽ các đoạn tín hiệu QRS đã cắt
for label, segments in labeled_segments.items():
    if label == 'N':
        for i, segment in enumerate(segments[:4000]):
            # plot_qrs_segment(segment, i, max_label)
            save_qrs_segment_as_image(segment, i, label)
