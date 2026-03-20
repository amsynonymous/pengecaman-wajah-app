import cv2
import numpy as np
from keras.models import load_model

# ======================
# LOAD MODELS
# ======================

# face
face_model = cv2.dnn.readNet(
    "opencv_face_detector_uint8.pb",
    "opencv_face_detector.pbtxt"
)

# gender
gender_model = cv2.dnn.readNet(
    "gender_net.caffemodel",
    "gender_deploy.prototxt"
)

# age
age_model = cv2.dnn.readNet(
    "age_net.caffemodel",
    "age_deploy.prototxt"
)

# emotion
emotion_model = load_model(
    "fer2013_mini_XCEPTION.110-0.65.hdf5",
    compile=False
)

# ======================
# LABEL
# ======================

gender_list = ["Male", "Female"]

age_list = [
    "(0-2)", "(4-6)", "(8-12)", "(15-20)",
    "(25-32)", "(38-43)", "(48-53)", "(60-100)"
]

emotion_labels = [
    "Angry", "Disgust", "Fear",
    "Happy", "Sad", "Surprise", "Neutral"
]

# ======================
# START CAMERA
# ======================

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()

    if not ret:
        print("tak dapat buka camera")
        break

    # flip (mirror)
    frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]

    # ======================
    # FACE DETECT
    # ======================
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300))
    face_model.setInput(blob)
    result = face_model.forward()

    for i in range(result.shape[2]):
        confidence = result[0, 0, i, 2]

        if confidence > 0.5:
            box = result[0, 0, i, 3:7] * [w, h, w, h]
            x1, y1, x2, y2 = box.astype(int)

            # avoid crash
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            # ======================
            # GENDER + AGE
            # ======================
            face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227))

            # gender
            gender_model.setInput(face_blob)
            gender_pred = gender_model.forward()
            gender = gender_list[gender_pred[0].argmax()]

            # age
            age_model.setInput(face_blob)
            age_pred = age_model.forward()
            age = age_list[age_pred[0].argmax()]

            # ======================
            # EMOTION
            # ======================
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_gray = gray[y1:y2, x1:x2]

            try:
                face_gray = cv2.resize(face_gray, (64, 64))
                face_gray = face_gray / 255.0
                face_gray = face_gray.reshape(1, 64, 64, 1)

                emotion_pred = emotion_model.predict(face_gray, verbose=0)
                emotion = emotion_labels[emotion_pred.argmax()]
            except:
                emotion = "?"

            # ======================
            # DISPLAY
            # ======================
            label = f"{gender} {age} {emotion}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)

    cv2.imshow("Pengecaman Wajah", frame)

    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()