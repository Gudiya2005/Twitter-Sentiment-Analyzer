# Twitter Sentiment Analyzer 🐦💬

This is a Machine Learning web app that predicts the **sentiment of tweets** (Positive 😊, Neutral 😐, or Negative 😠).  
It was built using **Natural Language Processing (NLP)** techniques and deployed with **Gradio + Hugging Face Spaces**.

## 📌 Features
- Preprocessing using NLTK (tokenization, stopwords removal, etc.)
- Trained and compared multiple models (Logistic Regression & Linear SVM)
- Final model: **Linear SVM**
- Visualized tweet sentiments using **WordCloud**
- Interactive web interface with emoji-based sentiment results

## 🛠️ Tech Stack
- Python
- NLTK
- Scikit-learn
- WordCloud
- Gradio
- Hugging Face Spaces

## 📊 Dataset
- **Twitter US Airline Sentiment Dataset**  
  Source: [Kaggle Dataset Link](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)

## 💡 How to Use
1. Enter a tweet in the input box.
2. Click **Submit**.
3. The model will analyze and show the predicted sentiment with an emoji.

## 🧠 Model Details
- **Model Used:** Linear SVM
- **Text Processing:** Lowercasing, Stopword Removal, Tokenization, TF-IDF Vectorization
- **Evaluation Metrics:** Accuracy, Confusion Matrix, Classification Report

## 📎 Files Included
- `app.py` – Gradio app
- `sentiment_model.pkl` – Trained SVM model
- `vectorizer.pkl` – TF-IDF vectorizer
- `requirements.txt` – List of required Python packages

## 📍 Live Demo

👉 [Try it on Hugging Face Spaces](https://huggingface.co/spaces/GudiyaSharma/twitter-sentiment-analyzer)

## 👩‍💻 Developed By

**Gudiya Kumari Sharma**
B.Tech CSE (AI & ML) | Arka Jain University  
