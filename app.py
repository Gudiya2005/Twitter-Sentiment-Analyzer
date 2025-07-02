import joblib
import gradio as gr
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_and_preprocess(text):
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    words = word_tokenize(text, preserve_line=True)
    filtered = [stemmer.stem(w) for w in words if w not in stop_words and w.isalpha()] 
    return " ".join(filtered)

def predict_sentiment(text):
    processed = clean_and_preprocess(text)
    vectorized = vectorizer.transform([processed])
    prediction = model.predict(vectorized)[0]

    if prediction == 0:
        return "Negative 😠"
    elif prediction == 1:
        return "Neutral 😐"
    else:
        return "Positive 😊"

interface = gr.Interface(
    fn=predict_sentiment,            
    inputs=gr.Textbox(
        lines=3,
        placeholder="Enter a tweet to analyse sentiment...",
        label="Tweet",
    ),
    outputs=gr.Textbox(label="Predicted Sentiment 😊😐😠"),
    title="Twitter Sentiment Analyzer",
    description="Enter a tweet and find out if it's Positive, Neutral, or Negative.🚀",
    theme=gr.themes.Default()
)

interface.launch(share=True)