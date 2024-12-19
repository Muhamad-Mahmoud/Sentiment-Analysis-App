import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

class TextProcessor:
    @staticmethod
    def clean_text(text):
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        text = re.sub(r'\W', ' ', text.lower())  # Remove non-word characters and convert to lowercase
        return ' '.join(
            lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words
        )
