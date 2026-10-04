# Movie Recommendation System

A content-based movie recommendation app that finds movies with similar metadata. It uses the TMDB 5000 Movies and Credits dataset, NLP preprocessing, and cosine similarity, and is available through a lightweight Streamlit interface.

## Features

- Searchable movie selector
- Up to five similar movie recommendations
- Case-insensitive title matching in the reusable recommendation module
- NLP preprocessing with NLTK Porter stemming
- CountVectorizer bag-of-words features and cosine similarity
- Precomputed sparse similarity artifacts, generated automatically if missing

## How It Works

Movie Metadata  
→ Feature Engineering (overview, genres, keywords, top three cast members, director)  
→ Combined Tags  
→ Lowercasing and Porter Stemming  
→ CountVectorizer (up to 5,000 features, English stop words removed)  
→ Movie Vectors  
→ Sparse Cosine Similarity  
→ Top-K Recommendations

The app compares the selected movie's metadata with every other movie. This is content-based filtering: recommendations are driven by item attributes rather than user ratings or viewing history.

## Machine Learning / NLP Concepts

- **Feature engineering:** combines plot overview, genres, keywords, cast, and director.
- **Text preprocessing:** turns metadata fields into a single searchable document per movie.
- **Stemming:** Porter stemming reduces related word forms to a common stem.
- **Bag of words:** CountVectorizer represents each movie by word occurrence counts.
- **Cosine similarity:** ranks movies by the angle between their feature vectors.
- **Content-based filtering:** uses movie attributes to find similar items.

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, NLTK, and Streamlit.

## Dataset

This project uses the **TMDB 5000 Movie Dataset** (movie and credits CSV files), distributed via Kaggle:
[TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).

The dataset is provided by its original authors and TMDB; this project does not claim ownership. Refer to the dataset page and TMDB terms for attribution and use requirements.

## Project Structure

```text
.
├── app.py
├── recommender.py
├── data.py
├── requirements.txt
├── README.md
├── .gitignore
├── artifacts/
│   ├── movies.pkl
│   └── vectors.pkl
├── dataset/
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
└── notebooks/
    └── main.ipynb
```

The notebook records the original exploration; the app does not execute it. Sparse CountVectorizer feature vectors are saved instead of the much larger all-pairs similarity matrix. Cosine similarity is calculated for the selected movie only, keeping the artifacts practical to store and load without changing the recommendation method. If artifacts are absent, the app builds them from the CSV files on startup. To generate them explicitly, run:

```bash
python data.py
```

Pickle files should only be loaded from trusted sources.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install the application dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run Locally

```bash
streamlit run app.py
```

## Deployment

Deploy the repository with [Streamlit Community Cloud](https://share.streamlit.io/), select `app.py` as the main file, and keep `requirements.txt` at the repository root. Ensure both CSV files and generated artifacts are committed; alternatively, the artifacts can be regenerated at app startup from the included dataset.

## Limitations

- Recommendations are content-based and do not use collaborative filtering.
- There is no personalized user history or rating model.
- Results depend on the available metadata and vocabulary.
- Similarity is based on word counts, not semantic embeddings.

## Future Improvements

- Semantic text embeddings or hybrid recommendation
- Collaborative filtering and preference modeling
- Better ranking and recommendation explanations
- Enriched movie metadata and user profiles
