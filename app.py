import streamlit as st
import fitz  # PyMuPDF
import re
import numpy as np
import pickle
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

model = load_model('resume_classifier_cnn.h5')

max_len = 300

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

def predict_category(text):
    processed = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([processed])
    padded = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(padded)
    pred_label = le.inverse_transform([np.argmax(pred)])
    return pred_label[0]

st.title("Resume Classifier")
uploaded_file = st.file_uploader("Upload a Resume (PDF)", type="pdf")

if uploaded_file is not None:
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()

    st.subheader("Predicted Category:")
    st.success(predict_category(text))
