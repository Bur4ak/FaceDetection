
import cv2

# CascadeClassifier'ı yüz tanıma için kullan
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Webcam başlat
video_capture = cv2.VideoCapture(0)

while True:
    # Webcam'den bir çerçeve al
    ret, frame = video_capture.read()
    
    # Çerçeveyi griye dönüştür (yüz tespiti için daha iyi)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Yüzleri tespit et
    faces = face_classifier.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Tespit edilen yüzleri dikdörtgen içine al
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Çerçeveyi göster
    cv2.imshow('Face Detection', frame)
    
    # Çıkış için 'q' tuşuna bas
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Webcam'i serbest bırak
video_capture.release()

# Pencereleri kapat
cv2.destroyAllWindows()




