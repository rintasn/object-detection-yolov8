import os
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

def ensure_directory(directory):
    """
    Membuat direktori jika belum ada
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def detect_objects(image_path, model_path='best.pt', conf_threshold=0.25):
    """
    Fungsi untuk melakukan deteksi objek menggunakan YOLOv8
    """
    # Memuat model yang telah dilatih
    model = YOLO(model_path)

    # Melakukan inferensi pada gambar dengan threshold confidence
    results = model.predict(
        source=image_path, 
        save=False, 
        conf=conf_threshold
    )

    # Menampilkan hasil deteksi dengan teks prediksi
    for result in results:
        img = result.orig_img
        boxes = result.boxes  # Dapatkan bounding boxes
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Koordinat bounding box
            conf = box.conf[0]  # Confidence score
            cls = box.cls[0]  # Kelas yang terdeteksi
            class_name = model.names[int(cls)]

            # Warna berbeda untuk setiap kelas
            color = (
                (255, 0, 0),   # Biru untuk kelas pertama
                (0, 255, 0),   # Hijau untuk kelas kedua
                (0, 0, 255)    # Merah untuk kelas ketiga
            )[int(cls) % 3]

            # Gambar bounding box
            cv2.rectangle(
                img, 
                (int(x1), int(y1)), 
                (int(x2), int(y2)), 
                color, 
                2
            )
            
            # Tambahkan teks label dan confidence
            text = f"{class_name} {conf:.2f}"
            cv2.putText(
                img, 
                text, 
                (int(x1), int(y1) - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                color, 
                2
            )

        return img, results

def save_detection_result(image, input_image_path, output_folder='detection_result'):
    """
    Menyimpan gambar hasil deteksi ke dalam folder tertentu
    """
    # Pastikan folder output ada
    ensure_directory(output_folder)
    
    # Dapatkan nama file asli
    filename = os.path.basename(input_image_path)
    
    # Buat path file output
    output_path = os.path.join(output_folder, filename)
    
    # Simpan gambar
    cv2.imwrite(output_path, image)
    print(f"Gambar hasil deteksi disimpan di {output_path}")

def main():
    # Path ke folder input gambar
    input_folder = 'datatest'
    
    # Path ke folder output
    output_folder = 'detection_result'
    
    # Path ke model
    model_path = 'best.pt'
    
    # Pastikan folder output ada
    ensure_directory(output_folder)
    
    # Proses setiap gambar di folder input
    for filename in os.listdir(input_folder):
        # Periksa apakah file adalah gambar
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Path lengkap gambar input
            input_image_path = os.path.join(input_folder, filename)
            
            # Lakukan deteksi objek
            try:
                detected_img, results = detect_objects(
                    input_image_path, 
                    model_path=model_path, 
                    conf_threshold=0.25
                )
                
                # Tampilkan jumlah objek yang terdeteksi
                print(f"Gambar: {filename}, Jumlah objek terdeteksi: {len(results[0].boxes)}")
                
                # Simpan gambar hasil deteksi
                save_detection_result(detected_img, input_image_path, output_folder)
            
            except Exception as e:
                print(f"Error memproses {filename}: {e}")

if __name__ == "__main__":
    main()