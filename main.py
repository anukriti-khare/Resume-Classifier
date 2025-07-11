from fastapi import FastAPI, File, UploadFile
import shutil
import os
import numpy as np
from tensorflow.keras.models import load_model
from utils import extract_text_from_pdf, preprocess_text, load_pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI()

model = load_model("resume_classifier_cnn.h5")
tokenizer = load_pickle("tokenizer.pkl")
label_encoder = load_pickle("label_encoder.pkl")

max_len = 300  

@app.post("/predict/")
async def predict_resume_category(file: UploadFile = File(...)):
    file_location = f"temp_{file.filename}"
    
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        text = extract_text_from_pdf(file_location)
        clean_text = preprocess_text(text)
        seq = tokenizer.texts_to_sequences([clean_text])
        padded = pad_sequences(seq, maxlen=max_len)

        pred = model.predict(padded)
        pred_label = label_encoder.inverse_transform([np.argmax(pred)])

        os.remove(file_location)

        return {
            "predicted_category": pred_label[0],
            "confidence": float(np.max(pred))
        }
    except Exception as e:
        return {"error": str(e)}
