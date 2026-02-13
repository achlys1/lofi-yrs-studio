import streamlit as st
import librosa
import numpy as np

st.set_page_config(page_title="AI Lofi Studio", page_icon="🎵")
st.title("🎵 AI Lofi Studio")
st.write("Upload a song to find the best Lofi loops!")

uploaded_file = st.file_uploader("Choose an MP3 file", type=["mp3"])

if uploaded_file:
    with st.spinner("AI is analyzing..."):
        y, sr = librosa.load(uploaded_file)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        st.metric(label="Detected Tempo", value=f"{round(float(tempo), 2)} BPM")
        
        # Stanza Finder logic
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        boundaries = librosa.segment.agglomerative(onset_env, 5)
        boundary_times = librosa.frames_to_time(boundaries, sr=sr)

        st.subheader("AI Identified Sections")
        for i in range(len(boundary_times)-1):
            st.info(f"Section {i+1}: {round(boundary_times[i], 2)}s to {round(boundary_times[i+1], 2)}s")
