# Real-Time ASL to Text Conversion System

## 1. Title Page

**Project Name:** Real-Time ASL to Text Conversion System  
**Prepared By:** [Your Name(s)]  
**University / Course:** [University Name, Department, Course Code]  
**Date:** [April 2026]

---

## 2. Abstract

This project presents a Real-Time ASL to Text Conversion System that translates hand gestures into readable text using a webcam.  
The system captures live video frames and detects hand landmarks using MediaPipe.  
These landmarks are processed as numerical features and passed to a machine learning model for gesture classification.  
Predicted labels are smoothed over multiple frames to improve stability and reduce misclassification noise.  
Detected gestures are converted into characters, words, and simple text output in real time.  
This work helps reduce communication barriers between deaf/mute users and non-sign-language users.

---

## 3. Problem Statement

Communication between deaf/mute individuals and people who do not understand sign language remains difficult in many daily situations.  
Most existing communication methods are either manual, slow, or unavailable in real-time interactions.  
There is a strong need for a practical system that can translate ASL gestures to text instantly, using affordable hardware.

---

## 4. Objectives

### Main Objective

Convert ASL gestures into text in real-time.

### Specific Objectives

- Detect hand landmarks accurately from webcam input.
- Classify gestures into ASL labels using an AI model.
- Build readable words/sentences from sequential gesture predictions.
- Improve prediction reliability with smoothing and confidence thresholds.

---

## 5. Technologies Used

- **Python** for system development.
- **OpenCV** for video capture, rendering, and user interface overlays.
- **MediaPipe** for real-time hand landmark detection (21 keypoints).
- **scikit-learn** for model training and classification.
- **NumPy / pandas / joblib** for feature handling, dataset management, and model persistence.

---

## 6. System Architecture

The overall processing flow is:

**Camera -> MediaPipe -> Hand Landmarks -> Trained Model -> Prediction -> Text Output**

1. Webcam captures live hand images.
2. MediaPipe detects hand and extracts landmark coordinates.
3. Landmarks are normalized into a feature vector.
4. The trained classifier predicts the most likely ASL label.
5. Smoothing logic stabilizes frame-by-frame predictions.
6. Stable predictions are appended to the output text stream.

---

## 7. Methodology

### 1. Data Collection

- A custom webcam script is used to collect gesture samples per label.
- For every capture, hand landmarks are extracted and saved.
- Collected features are stored in `data/landmarks.csv`.

### 2. Feature Extraction

- MediaPipe returns **21 hand points**, each with **(x, y, z)**.
- This gives **63 values** per sample.
- Features are normalized relative to the wrist and hand scale for consistency.

### 3. Model Training

- Landmark CSV data is split into training and validation sets.
- Features are standardized with `StandardScaler`.
- A scikit-learn classifier is trained on landmark vectors.
- In this implementation, an **MLPClassifier** is used for training and saved with joblib.
- The model and class labels are stored in the `models/` directory.

### 4. Real-Time Prediction

- Webcam frames are processed continuously.
- Each valid hand frame is converted to features and classified.
- Majority-vote smoothing is applied to reduce flicker/noise.
- Stable labels are committed to text, with support for **SPACE** and **DEL** actions.

---

## 8. Results

- The trained model produced high validation performance on the collected dataset (project console prints validation accuracy during training).
- The real-time demo successfully detects gestures and converts them into text on-screen.
- Built-in smoothing and confidence thresholds improved output stability in live conditions.

### Demo Observation

- Live camera feed shows hand landmarks and current predicted label.
- Text accumulates in real-time with manual controls (clear, backspace, save).

### Screenshots (Very Important)

Insert the following screenshots in the final submitted version:

1. Data collection window showing landmarks and current label.
2. Training terminal output showing validation accuracy and report.
3. Real-time prediction window with generated text.
4. Example saved output text file/log.

---

## 9. Challenges

- **Left hand vs right hand variation:** Different orientations can affect consistency.
- **Lighting and camera angle sensitivity:** Poor lighting lowers landmark reliability.
- **Gesture similarity:** Some signs are visually close, causing confusion.
- **User-dependent signing style:** Different users produce variation in hand shape and motion.

---

## 10. Future Work

- Extend support to full A-Z with larger and more diverse datasets.
- Add language modeling and word prediction for faster sentence creation.
- Integrate text-to-speech for voice output.
- Build a mobile/web application for wider accessibility.
- Explore sequence models (LSTM/Transformer) for dynamic ASL signs.

---

## 11. Conclusion

The Real-Time ASL to Text Conversion System demonstrates that low-cost camera input plus AI-based hand landmark analysis can provide practical sign-to-text translation.  
The prototype successfully performs real-time gesture recognition and text generation, reducing communication barriers.  
With larger datasets, improved models, and multimodal outputs, the system can become even more accurate and impactful.

---

## 12. References

1. MediaPipe Documentation - Hand Tracking: https://developers.google.com/mediapipe
2. OpenCV Documentation: https://docs.opencv.org
3. scikit-learn Documentation: https://scikit-learn.org/stable/
4. Python Documentation: https://docs.python.org/3/
5. Any additional tutorials, articles, or videos used during implementation (add citations as needed).
