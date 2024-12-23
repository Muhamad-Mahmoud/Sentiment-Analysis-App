from flask import Flask, render_template, request
import joblib
from text_processor import TextProcessor

app = Flask(__name__)

# Load the trained model
model = joblib.load('pipeline.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['text']
        cleaned_input = TextProcessor.preprocess(user_input)
        prediction = model.predict([cleaned_input])
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        
        return render_template(
            'index.html', 
            prediction_text=f'{sentiment}', 
            cleaned_text=f'{cleaned_input}',
            text_input=f'{user_input}'
        )

if __name__ == '__main__':
    app.run(debug=True)
