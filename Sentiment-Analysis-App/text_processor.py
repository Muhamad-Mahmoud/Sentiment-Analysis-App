from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

class TextProcessor:
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    @classmethod
    def preprocess(cls, text):
        # Remove HTML tags
        text = BeautifulSoup(text, 'html.parser').get_text()
        # Remove all non-word characters and digits
        text = re.sub(r'\W', ' ', text).lower()
        # Tokenize, lemmatize, and remove stopwords
        tokens = [cls.lemmatizer.lemmatize(word) for word in text.split() if word not in cls.stop_words]
        return ' '.join(tokens)
