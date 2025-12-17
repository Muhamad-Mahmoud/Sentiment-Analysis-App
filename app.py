from flask import Flask, render_template, request
import joblib
import os
import re

def clean_text(text: str) -> str:
    contractions = {
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "won't": "will not",
        "can't": "can not",
        "cannot": "can not",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "haven't": "have not",
        "hasn't": "has not",
        "hadn't": "had not",
        "wouldn't": "would not",
        "shouldn't": "should not",
        "couldn't": "could not",
    }

    text = text.lower()
    for c, e in contractions.items():
        text = text.replace(c, e)

    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


app = Flask(__name__)

MODEL_PATH = 'pipeline_bidirectional.pkl'
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form.get('text', '')

    if not user_input.strip():
        return render_template('index.html', prediction_text="Please enter some text.")

    cleaned_input = clean_text(user_input)

    if model is None:
        sentiment = "Model not loaded"
    else:
        prediction = model.predict([cleaned_input])
        sentiment = "Positive" if prediction[0] == 1 else "Negative"

    return render_template(
        'index.html',
        prediction_text=sentiment,
        cleaned_text=cleaned_input,
        text_input=user_input
    )


if __name__ == '__main__':
    app.run(debug=True)
