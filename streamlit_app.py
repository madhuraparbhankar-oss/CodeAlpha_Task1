import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import plotly.graph_objects as go
import os

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CSS Styling ----------------
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

/* Hide default file uploader style */
section[data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

section[data-testid="stFileUploader"] > div {
    background: transparent !important;
    border: none !important;
}

div[data-testid="stFileUploadDropzone"] {
    background: rgba(255, 255, 255, 0.1) !important;
    border: 2px dashed rgba(255, 255, 255, 0.5) !important;
    border-radius: 15px !important;
    backdrop-filter: blur(10px);
}

div[data-testid="stFileUploadDropzone"]:hover {
    background: rgba(255, 255, 255, 0.2) !important;
    border-color: rgba(255, 255, 255, 0.8) !important;
}

h1 {
    color: white !important;
    text-align: center;
    font-size: 3.5em !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    margin-bottom: 10px !important;
}
.subtitle {
    color: white;
    text-align: center;
    font-size: 1.3em;
    margin-bottom: 20px;
    opacity: 0.9;
}

/* Upload Card */
.upload-card {
    background: transparent !important;
    backdrop-filter: none !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 20px !important;
    text-align: center;
    box-shadow: none !important;
    margin: 10px 0 !important;
}
.upload-title {
    color: white;
    font-size: 1.8em;
    font-weight: bold;
    margin-bottom: 20px;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
}

/* Result Card */
.result-main-card {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    border-radius: 30px;
    padding: 50px;
    text-align: center;
    box-shadow: 0 15px 50px rgba(0,0,0,0.3);
    margin: 30px 0;
    transform: scale(1);
    transition: transform 0.3s;
}
.result-main-card:hover {
    transform: scale(1.02);
}
.emotion-display {
    font-size: 8em;
    margin: 20px 0;
    animation: bounce 1s ease;
}
@keyframes bounce {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-20px); }
}
.emotion-text {
    color: white;
    font-size: 4em;
    font-weight: bold;
    text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
    margin: 20px 0;
    letter-spacing: 3px;
}
.confidence-text {
    color: white;
    font-size: 2em;
    font-weight: 500;
    margin-top: 10px;
    opacity: 0.95;
}

/* Metrics Cards */
.metric-container {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(10px);
    border: 2px solid rgba(255, 255, 255, 0.3);
    border-radius: 20px;
    padding: 30px;
    margin: 20px 0;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}
.metric-title {
    color: white;
    font-size: 1.5em;
    font-weight: bold;
    margin-bottom: 20px;
    text-align: center;
}

/* Chart Container */
.chart-container {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    border: 2px solid rgba(255, 255, 255, 0.3);
    border-radius: 25px;
    padding: 30px;
    margin: 30px 0;
    box-shadow: 0 10px 40px rgba(0,0,0,0.2);
}
.stButton>button {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
    border: none;
    border-radius: 30px;
    padding: 20px 60px;
    font-size: 1.3em;
    font-weight: bold;
    box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    transition: all 0.3s;
    width: 100%;
    margin-top: 20px;
}
.stButton>button:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 35px rgba(0,0,0,0.4);
}

/* Audio player styling */
audio {
    width: 100%;
    margin: 20px 0;
    border-radius: 15px;
}

div[data-testid="stMetricValue"] {
    font-size: 1.8em;
    color: white;
    font-weight: bold;
}
div[data-testid="stMetricLabel"] {
    color: rgba(255, 255, 255, 0.9);
    font-size: 1.1em;
}
div[data-testid="stMetricDelta"] {
    color: rgba(255, 255, 255, 0.8);
}
</style>
""", unsafe_allow_html=True)
# ---------------- Emotion Mappings ----------------
emotion_emojis = {
    'angry': '😠', 'disgust': '🤢', 'fear': '😨', 'happy': '😄',
    'neutral': '😐', 'sad': '😢', 'surprise': '😲'
}

emotion_colors = {
    'angry': '#FF4444', 'disgust': '#AA00FF', 'fear': '#0099FF',
    'happy': '#FFD700', 'neutral': '#808080', 'sad': '#4169E1', 'surprise': '#FF6B35'
}

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# ---------------- Load Model ----------------
@st.cache_resource
def load_emotion_model():
    model_path = os.path.join(os.getcwd(), "emotion_model.h5")
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()
    return tf.keras.models.load_model(model_path)

model = load_emotion_model()

# ---------------- Feature Extraction ----------------
def extract_features(audio_file):
    data, sr = librosa.load(audio_file, duration=3, offset=0.5)
    mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs

# ---------------- Confidence Chart ----------------
def create_confidence_chart(prediction, emotion_labels):
    fig = go.Figure()
    colors = [emotion_colors[label] for label in emotion_labels]
    fig.add_trace(go.Bar(
        y=emotion_labels,
        x=prediction[0]*100,
        orientation='h',
        marker=dict(color=colors, line=dict(color='rgba(255,255,255,0.6)', width=2)),
        text=[f'{val*100:.1f}%' for val in prediction[0]],
        textposition='outside',
        textfont=dict(color="#F2F1E9", size=14, family='Arial Black'),
        hovertemplate='<b>%{y}</b><br>Confidence: %{x:.2f}%<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text="Confidence Levels for All Emotions", font=dict(color='white', size=20, family='Arial Black')),
        xaxis_title="Confidence (%)",
        yaxis_title="Emotions",
        height=450,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=15),
        xaxis=dict(gridcolor='rgba(255,255,255,0.2)', range=[0,105]),
        yaxis=dict(gridcolor='rgba(255,255,255,0.2)')
    )
    return fig

# ---------------- Header ----------------
st.markdown("<h1>🎭 Speech Emotion Recognition</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:white; text-align:center; font-size:1.3em;'>Analyze emotions from voice using AI-powered deep learning</p>", unsafe_allow_html=True)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("### 📊 About")
    st.info("""
This AI-powered application analyzes speech audio and detects emotions using deep learning.

**Supported Emotions:**
- 😠 Angry
- 🤢 Disgust
- 😨 Fear
- 😄 Happy
- 😐 Neutral
- 😢 Sad
- 😲 Surprise
""")
    st.markdown("### 🎯 How to Use")
    st.markdown("1. Upload a WAV audio file\n2. Click 'Analyze Emotion'\n3. View detailed results")
    st.markdown("### ⚙️ Model Info")
    st.markdown(f"- **Model Type:** Deep Neural Network\n- **Emotions:** {len(emotion_labels)}\n- **Status:** ✅ Ready")

# ---------------- Upload Section ----------------
uploaded_file = st.file_uploader("🎤 Upload Your Audio File (WAV only)", type=["wav"])
if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    if st.button("🔍 Analyze Emotion"):
        with st.spinner("🤖 Analyzing emotion..."):
            with open("temp.wav", "wb") as f:
                f.write(uploaded_file.read())
            features = extract_features("temp.wav")
            features = np.expand_dims(features, axis=0)
            prediction = model.predict(features, verbose=0)
            predicted_emotion = emotion_labels[np.argmax(prediction)]
            confidence = np.max(prediction)*100

            # Display results
            st.success(f"✅ Detected Emotion: {predicted_emotion.upper()} ({confidence:.2f}%)")
            st.plotly_chart(create_confidence_chart(prediction, emotion_labels), use_container_width=True)

