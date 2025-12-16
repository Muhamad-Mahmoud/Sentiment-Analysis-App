from flask import Flask, render_template, request
import joblib
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import re
from typing import List, Union, Any

def clean_text(text: str) -> str:
    """
    Clean text using the same preprocessing steps from model training.
    This matches the preprocessing in RNN_Project.ipynb:
    1. Convert to lowercase
    2. Expand contractions (e.g., "don't" -> "do not")
    3. Remove HTML tags
    4. Remove special characters (keep only letters)
    5. Remove extra spaces
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned text
    """
    # Define common contractions
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
        "mightn't": "might not",
        "mustn't": "must not",
        "i'm": "i am",
        "you're": "you are",
        "he's": "he is",
        "she's": "she is",
        "it's": "it is",
        "we're": "we are",
        "they're": "they are",
        "i've": "i have",
        "you've": "you have",
        "we've": "we have",
        "they've": "they have",
        "i'll": "i will",
        "you'll": "you will",
        "he'll": "he will",
        "she'll": "she will",
        "we'll": "we will",
        "they'll": "they will",
    }
    
    text = text.lower()
    
    # Expand contractions BEFORE removing special characters
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    text = re.sub(r"<.*?>", " ", text)          # remove HTML tags
    text = re.sub(r"[^a-zA-Z]", " ", text)      # remove special chars
    text = re.sub(r"\s+", " ", text).strip()    # remove double spaces
    return text

class SentimentPipeline:
    """
    A pipeline class to handle tokenization, padding, and prediction for the Keras model.
    This class mimics the structure expected by the pickled object.
    """
    def __init__(self, model: Any, tokenizer: Any, max_len: int):
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len

    def predict(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Predicts sentiment for valid input text(s).
        
        Args:
            texts (Union[str, List[str]]): Input text or list of texts.
            
        Returns:
            np.ndarray: Array of binary predictions (1 for Positive, 0 for Negative).
        """
        if isinstance(texts, str):
            texts = [texts]

        seqs = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(seqs, maxlen=self.max_len, padding="post")

        preds = self.model.predict(padded)
        return (preds > 0.5).astype(int).ravel()

app = Flask(__name__)

# --- Model Loading ---
MODEL_PATH = 'pipeline_bidirectional.pkl'
model = None

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
        print(f"✅ Model type: {type(model)}")
        print(f"✅ Model has predict method: {hasattr(model, 'predict')}")
    else:
        print(f"❌ Warning: Model file not found at {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()


@app.route('/')
def home() -> str:
    """Renders the home page."""
    return render_template('index.html')

@app.route('/about')
def about() -> str:
    """Renders the about page."""
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict() -> str:
    """
    Handles the prediction request.
    Preprocesses the input text, runs the model, and renders the result.
    """
    if request.method == 'POST':
        user_input = request.form.get('text', '')
        
        if not user_input.strip():
             return render_template('index.html', prediction_text="Please enter some text.")
        
        # Clean the text using the same preprocessing as training
        cleaned_input = clean_text(user_input)
        
        sentiment = "Unknown"
        print(f"🔍 DEBUG: model is None? {model is None}")
        print(f"🔍 DEBUG: model type: {type(model) if model else 'None'}")
        
        if model is not None:
            try:
                print(f"🔍 DEBUG: Calling model.predict with: {cleaned_input[:50]}...")
                prediction = model.predict([cleaned_input])
                print(f"🔍 DEBUG: Prediction result: {prediction}")
                sentiment = "Positive" if prediction[0] == 1 else "Negative"
            except Exception as e:
                sentiment = f"Error during prediction: {str(e)}"
                print(f"❌ Prediction error: {e}")
                import traceback
                traceback.print_exc()
        else:
            sentiment = "Model not loaded"
            print(f"❌ Model is None!")

        return render_template(
            'index.html', 
            prediction_text=f'{sentiment}', 
            cleaned_text=f'{cleaned_input}',
            text_input=f'{user_input}'
        )
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
