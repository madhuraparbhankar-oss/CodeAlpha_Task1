The Emotion Speech Recognition project aims to identify human emotions from audio speech using machine learning and deep learning techniques.
By analyzing vocal tone, pitch, and energy levels, the system classifies emotions such as Happy, Sad, Angry, Neutral, and more.
It demonstrates the power of audio signal processing combined with neural networks for affective computing applications.

🎯 Objectives

Recognize emotional states from recorded speech.

Understand relationships between vocal features and emotions.

Build an end-to-end ML/DL pipeline from feature extraction to emotion prediction.

⚙️ Features

🎤 Audio Feature Extraction using MFCC, Chroma, and Spectral features.

🤖 ML & DL Models — supports SVM, Random Forest, CNN, or LSTM.

📈 Model Training & Evaluation with accuracy and confusion matrix visualization.

🧪 Real-time Prediction for testing emotions using microphone input.

📊 Visualization of feature distributions and training performance.

🧩 Tech Stack

Language: Python

Libraries & Tools:

librosa — for audio feature extraction

numpy, pandas — for data handling

matplotlib, seaborn — for visualization

scikit-learn — for machine learning models

tensorflow / keras — for deep learning models

🧠 Dataset

The model can be trained on publicly available datasets such as:

RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

TESS (Toronto Emotional Speech Set)

CREMA-D (Crowd-Sourced Emotional Multimodal Actors Dataset)

Each dataset includes audio files labeled with different emotional categories.

🚀 How It Works

Preprocessing: Load and clean audio data.

Feature Extraction: Extract MFCCs and other key acoustic features using librosa.

Model Training: Train a CNN or LSTM model on extracted features.

Evaluation: Validate model using accuracy, precision, and confusion matrix.

Prediction: Classify emotion from new or real-time audio input.

📊 Example Emotions
Emotion	Label
Happy	😀
Sad	😢
Angry	😠
Fearful	😨
Neutral	😐
Disgust	🤢
Surprise	😲


To run app.py from VS code run this command 
 streamlit run your_path/app.py


