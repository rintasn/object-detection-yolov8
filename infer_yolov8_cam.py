from ultralytics import YOLO
import cv2

# Memuat model yang telah dilatih
model = YOLO('yolov8n.pt')  # Ganti dengan path ke model Anda

# Buka webcam (0 untuk webcam default)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam tidak dapat diakses.")
    exit()

while True:
    # Baca frame dari webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Tidak dapat membaca frame dari webcam.")
        break

    # Melakukan inferensi pada frame dengan threshold confidence yang lebih rendah
    results = model.predict(source=frame, save=False, conf=0.25)  # Menggunakan frame sebagai input

    # Menampilkan hasil deteksi dengan teks prediksi
    for result in results:
        img = result.orig_img
        boxes = result.boxes  # Dapatkan bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Koordinat bounding box
            conf = box.conf[0]  # Confidence score
            cls = box.cls[0]  # Kelas yang terdeteksi
            class_name = model.names[int(cls)]

            # Gambar bounding box
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            text = f"{class_name} {conf:.2f}"
            cv2.putText(img, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Resize image for display
    img_resized = cv2.resize(img, (800, 600))  # Ubah ukuran sesuai kebutuhan

    # Tampilkan hasil deteksi
    cv2.imshow("Deteksi Objek", img_resized)

    # Tekan 'q' untuk keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan webcam dan tutup jendela
cap.release()
cv2.destroyAllWindows()
