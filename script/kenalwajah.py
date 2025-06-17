import cv2, time
from PIL import Image

camera = 0
video = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
a = 0

recognizer = cv2.face.LBPHFaceRecognizer_create()
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer.read('D:/SEMESTER 6/AI for DS/face_recognition/training/training.xml')

id = 0
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (255, 0, 0)

while True:
    check, frame = video.read()
    print(check)
    print(frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        a = a+1
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        id, conf = recognizer.predict(gray[y:y+h, x:x+w])

        if id == 1:
            id = "Ridho"
        elif (id==2):
            id = "Fadil"

        cv2.putText(frame, str(id), (x + w, y + h), fontFace, fontScale, fontColor)

    cv2.imshow("wajah", frame)

    if cv2.waitKey(1) == ord('q'):
        break

    print(a)

video.release()
cv2.destroyAllWindows()