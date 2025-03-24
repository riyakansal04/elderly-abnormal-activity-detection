# Elderly-Abnormal-Activity-Detection
This project uses deep learning and machine learning models to monitor elderly individuals through CCTV footage. It detects abnormal activities such as falls, strokes, and other health events using pose estimation and classification. The system triggers real-time alerts via email for immediate intervention.
Overview

This project is a real-time elderly fall and abnormal activity detection system using CCTV footage. The system leverages MediaPipe Pose Estimation and a CNN-LSTM deep learning model to detect activities such as falls, strokes, and other health events. If an abnormal activity is detected for more than the threshold, an email alert is automatically sent to notify caregivers.

Features

✅ Real-time pose detection using MediaPipe
✅ Activity classification using deep learning (CNN-LSTM)
✅ Email alerts for abnormal activities (fall, heart stroke, headache, cough, etc.)
✅ Web-based interface for live video streaming
✅ Secure and efficient real-time video processing

📂 Elderly_Activity_Detection
│── app.py                 # Main Flask application
│── activity_classifier_optimized.h5  # Trained deep learning model
│── scaler.pkl             # StandardScaler model for preprocessing
│── label_encoder.pkl      # Label Encoder for activity classes
│── templates
│   ├── index.html         # Web interface for video streaming
│── static
│   ├── styles.css         # Frontend styling
└── README.md              # Project documentation

**Set-UP file:**
1.Clone the project files into the desktop folder of your system.

2.Open Command Prompt from the project folder and activate the environment using: conda activate human.

3.Run python training.py to collect dataset samples. Click on the camera feed, press 's' to save a pose with a label, or 'q' to stop data collection.

4.Open and execute each cell in train.ipynb to train the model with the collected data.

5.Start the application by running: python app.py in the command prompt.

6.Open localhost:5000 in a web browser to view the results.
