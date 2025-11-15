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
.main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
.stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }

/* Add your previous CSS styling here */
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
