# 🎬 Movie Recommendation System

A content-based movie recommendation system built with **Python, NLP, Scikit-learn, and Streamlit**.

The system recommends movies based on their similarity using movie metadata such as **genres, keywords, overview, cast, and director**.

## 🚀 Features

- Content-based movie recommendations
- NLP preprocessing with Porter Stemmer
- CountVectorizer for feature extraction
- Cosine Similarity for movie matching
- Top 5 recommendations
- Interactive Streamlit UI
- Precomputed artifacts for faster inference

## 🧠 Pipeline

```text
Movie Metadata
      ↓
Feature Engineering
      ↓
Tags Creation
      ↓
Porter Stemming
      ↓
CountVectorizer
      ↓
Cosine Similarity
      ↓
Top 5 Recommendations
```

## 🛠️ Tech Stack

**Python · Pandas · NumPy · Scikit-learn · NLTK · Streamlit**

## 📂 Structure

```text
movie-recommendation-system/
├── app.py
├── recommender.py
├── data.py
├── requirements.txt
├── artifacts/
├── dataset/
└── notebooks/
    └── main.ipynb
```

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🌐 Live Demo

https://movie-recommendation-system-adxsd.streamlit.app/

## 📊 Dataset

TMDB 5000 Movies and Credits Dataset.

## 🔮 Future Improvements

- Semantic embeddings
- Hybrid recommendation
- Collaborative filtering
- Personalized recommendations

## 👨‍💻 Author

**Aditya Narayan**  
B.Tech CSE — IIIT Bhubaneswar

