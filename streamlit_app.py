import streamlit as st
print("App started")
import numpy as np
from utils import extract_text_from_pdf, preprocess_text, load_pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("resume_classifier_cnn.h5")
tokenizer = load_pickle("tokenizer.pkl")
label_encoder = load_pickle("label_encoder.pkl")
max_len = 300

st.set_page_config(page_title="Resume Classifier", layout="centered")
st.title("Resume Job Role Classifier")
st.markdown("Upload a **PDF Resume** to predict the job role.")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    with open("temp_resume.pdf", "wb") as f:
        f.write(uploaded_file.read())

    try:
        text = extract_text_from_pdf("temp_resume.pdf")
        clean_text = preprocess_text(text)
        seq = tokenizer.texts_to_sequences([clean_text])
        padded = pad_sequences(seq, maxlen=max_len)

        pred = model.predict(padded)
        pred_label = label_encoder.inverse_transform([np.argmax(pred)])
        confidence = float(np.max(pred))

        st.success(f"Predicted Category: **{pred_label[0]}**")
        st.info(f"Confidence: **{confidence:.2f}**")

    except Exception as e:
        st.error(f"Error: {e}")
