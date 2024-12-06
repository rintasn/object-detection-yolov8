import os
from ultralytics import YOLO

def main():
    # Tentukan path ke dataset dan model
    data_yaml = 'dataset/data.yaml'  # Path ke file konfigurasi dataset
    model_path = 'yolov8n.pt'  # Model YOLOv8 yang akan digunakan
    epochs = 200  # Jumlah epoch untuk pelatihan
    img_size = 640  # Ukuran gambar

    # Melatih model
    model = YOLO(model_path)  # Memuat model
    model.train(data=data_yaml, epochs=epochs, imgsz=img_size, workers=2, device='0')  # Gunakan GPU pertama

if __name__ == "__main__":
    main()
