import joblib
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from PIL import Image
import numpy as np

MODEL_PATH = "fake_news_model.pkl"
VEC_PATH = "vectorizer.pkl"

# Train the model if not already saved
if not os.path.exists(MODEL_PATH) or not os.path.exists(VEC_PATH):
    data = {
        "text": [
            "Aliens invaded Earth!",
            "Apple launched new iPhone.",
            "Vaccines turn people into zombies.",
            "NASA confirms flat Earth theory.",
            "Water is essential for life."
        ],
        "label": [1, 0, 1, 1, 0]  # 1 = Fake, 0 = Real
    }
    df = pd.DataFrame(data)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["text"])
    y = df["label"]

    model = LogisticRegression()
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VEC_PATH)
else:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)

def detect_fake_news(text):
    """
    Detects whether a given news text is fake or real.
    """
    x_test = vectorizer.transform([text])
    prediction = model.predict(x_test)[0]
    confidence = model.predict_proba(x_test).max()
    label = "Fake News" if prediction == 1 else "Real News"
    return f"{label} (Confidence: {confidence * 100:.2f}%)"

def detect_ai_media(filepath):
    """
    Detects whether an image might be AI-generated based on pixel variance.
    """
    if filepath.lower().endswith(('.jpg', '.jpeg', '.png')):
        try:
            img = Image.open(filepath).convert('RGB')
            img = img.resize((128, 128))
            img_np = np.array(img)
            variance = np.var(img_np)
            if variance < 100:
                return "Possibly AI-Generated Image"
            else:
                return "Likely Real Image"
        except:
            return "Error processing image"
    elif filepath.lower().endswith(('.mp4', '.avi', '.mov')):
        return "Video detection not implemented"
    else:
        return "Unsupported file type"
