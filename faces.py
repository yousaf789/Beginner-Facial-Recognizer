import numpy as np
import cv2
import pickle

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {}
with open("labels.pickle", "rb") as f:
    orig_labels = pickle.load(f)
    labels = {v:k for k,v in orig_labels.items()} # inverting labels with their value


while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        id_, conf = recognizer.predict(roi_gray)
        if conf >= 65:
            # print(id_)
            # print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            cv2.putText(frame, name, (x+w, y+h), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        img_item = "My image.jpeg"

        cv2.imwrite(img_item, roi_gray)


    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):

        break

cap.release()
cv2.destroyAllWindows()
