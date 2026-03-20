# 🎥 AI Face Detection (Gender, Age & Emotion)

This is a simple Python project that uses OpenCV and Deep Learning models to detect:

* 👤 Face
* 👦 Gender
* 🎂 Age
* 😄 Emotion

All detections are done in **real-time using webcam**.

---

## 🚀 Features

* Real-time face detection
* Gender prediction (Male/Female)
* Age estimation (range)
* Emotion detection (Happy, Sad, Angry, etc.)
* Simple and beginner-friendly code

---

## 🛠️ Technologies Used

* Python
* OpenCV
* TensorFlow / Keras
* Pre-trained Deep Learning Models

---

## 📦 Requirements

Install dependencies:

```bash
pip install opencv-python numpy tensorflow keras
```

---

## 📁 Required Files

Make sure these files are in the same folder as `app.py`:

```
opencv_face_detector_uint8.pb
opencv_face_detector.pbtxt

gender_net.caffemodel
gender_deploy.prototxt

age_net.caffemodel
age_deploy.prototxt

fer2013_mini_XCEPTION.102-0.66.hdf5
```

---

## ▶️ How to Run

```bash
python app.py
```

Press **Q** to exit.

---

## 🧠 How It Works

1. Detect face using OpenCV DNN
2. Extract face region
3. Predict:

   * Gender
   * Age
   * Emotion
4. Display results on screen

---

## ⚠️ Notes

* Age is estimated in ranges (not exact)
* Emotion detection may not be 100% accurate
* Lighting and camera quality affect results

---

## 📸 Demo

(Add screenshot here)

---

## 📌 Future Improvements

* Better accuracy using advanced models
* GUI interface (Streamlit / Tkinter)
* Save detection results
* Convert to desktop app (.exe)

---

## 👨‍💻 Author

Your Name Here

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
