"""Build and query the content-based movie recommendation model."""

from __future__ import annotations

import ast
import pickle
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = PROJECT_DIR / "dataset"
DEFAULT_ARTIFACT_DIR = PROJECT_DIR / "artifacts"
MOVIES_ARTIFACT = "movies.pkl"
VECTORS_ARTIFACT = "vectors.pkl"
RECOMMENDATION_COUNT = 5


def _parse_metadata(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def _named_values(value: Any, limit: int | None = None) -> list[str]:
    items = _parse_metadata(value)
    names = [item["name"] for item in items if isinstance(item, dict) and item.get("name")]
    if limit is not None:
        names = names[:limit]
    return [str(name).replace(" ", "") for name in names]


def _director(value: Any) -> list[str]:
    for item in _parse_metadata(value):
        if isinstance(item, dict) and item.get("job") == "Director" and item.get("name"):
            return [str(item["name"]).replace(" ", "")]
    return []


def preprocess_movies(
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
) -> pd.DataFrame:
    """Recreate the notebook's movie tags from the two TMDB CSV files."""
    dataset_dir = Path(dataset_dir)
    movies_path = dataset_dir / "tmdb_5000_movies.csv"
    credits_path = dataset_dir / "tmdb_5000_credits.csv"
    missing = [path.name for path in (movies_path, credits_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Required dataset file(s) not found in {dataset_dir}: {', '.join(missing)}"
        )

    movies_df = pd.read_csv(movies_path)
    credits_df = pd.read_csv(credits_path)
    movies = movies_df.merge(
        credits_df,
        left_on="id",
        right_on="movie_id",
        validate="one_to_one",
    )
    movies = movies[["genres", "id", "keywords", "title", "overview", "cast", "crew"]]
    movies = movies.dropna(subset=["id", "title", "overview"]).copy()
    movies = movies.rename(columns={"id": "movie_id"})

    movies["genres"] = movies["genres"].apply(_named_values)
    movies["keywords"] = movies["keywords"].apply(_named_values)
    movies["cast"] = movies["cast"].apply(lambda value: _named_values(value, limit=3))
    movies["crew"] = movies["crew"].apply(_director)
    movies["overview"] = movies["overview"].astype(str).apply(str.split)

    tag_columns = ["overview", "genres", "keywords", "crew", "cast"]
    movies["tags"] = movies[tag_columns].apply(
        lambda row: " ".join(word for values in row for word in values).lower(),
        axis=1,
    )
    movies["tags"] = movies["tags"].apply(stem_text)
    return movies[["movie_id", "title", "tags"]].reset_index(drop=True)


_STEMMER = PorterStemmer()


def stem_text(text: str) -> str:
    return " ".join(_STEMMER.stem(word) for word in text.split())


def build_artifacts(
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
) -> tuple[pd.DataFrame, Any]:
    """Build and persist movie metadata and sparse CountVectorizer features."""
    movies = preprocess_movies(dataset_dir)
    vectorizer = CountVectorizer(max_features=5000, stop_words="english")
    vectors = vectorizer.fit_transform(movies["tags"])
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with (artifact_dir / MOVIES_ARTIFACT).open("wb") as file:
        pickle.dump(movies, file, protocol=pickle.HIGHEST_PROTOCOL)
    with (artifact_dir / VECTORS_ARTIFACT).open("wb") as file:
        pickle.dump(vectors, file, protocol=pickle.HIGHEST_PROTOCOL)
    return movies, vectors


@dataclass
class MovieRecommender:
    movies: pd.DataFrame
    vectors: Any

    @classmethod
    def from_artifacts(
        cls,
        artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
        dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    ) -> "MovieRecommender":
        artifact_dir = Path(artifact_dir)
        movies_path = artifact_dir / MOVIES_ARTIFACT
        vectors_path = artifact_dir / VECTORS_ARTIFACT
        if not movies_path.is_file() or not vectors_path.is_file():
            build_artifacts(dataset_dir, artifact_dir)
        with movies_path.open("rb") as file:
            movies = pickle.load(file)
        with vectors_path.open("rb") as file:
            vectors = pickle.load(file)
        if vectors.shape[0] != len(movies):
            raise ValueError("Movie and feature-vector artifacts have incompatible dimensions.")
        return cls(movies=movies, vectors=vectors)

    @property
    def titles(self) -> list[str]:
        return self.movies["title"].dropna().astype(str).drop_duplicates().tolist()

    def recommend(
        self,
        movie_title: str,
        n: int = RECOMMENDATION_COUNT,
    ) -> list[dict[str, Any]]:
        """Return up to ``n`` similar movies, or an empty list if no title matches."""
        if not isinstance(movie_title, str) or not movie_title.strip() or n <= 0:
            return []

        query = re.sub(r"\s+", " ", movie_title.strip()).casefold()
        title_keys = self.movies["title"].astype(str).str.strip().str.casefold()
        matches = self.movies.index[title_keys == query]
        if matches.empty:
            return []
        movie_index = int(matches[0])

        scores = cosine_similarity(
            self.vectors[movie_index],
            self.vectors,
        ).ravel()
        same_title = title_keys.to_numpy() == query
        scores[same_title] = -1
        scores[movie_index] = -1
        ordered_indices = scores.argsort()[::-1]

        results = []
        for index in ordered_indices:
            if scores[index] < 0:
                break
            row = self.movies.iloc[int(index)]
            results.append(
                {
                    "title": str(row["title"]),
                    "movie_id": int(row["movie_id"]),
                    "similarity": float(scores[index]),
                }
            )
            if len(results) == n:
                break
        return results


def recommend(
    movie_title: str,
    n: int = RECOMMENDATION_COUNT,
    recommender: MovieRecommender | None = None,
) -> list[dict[str, Any]]:
    """Convenience API; pass a loaded model to reuse it across recommendations."""
    model_instance = recommender or _default_recommender()
    return model_instance.recommend(movie_title, n=n)


@lru_cache(maxsize=1)
def _default_recommender() -> MovieRecommender:
    return MovieRecommender.from_artifacts()


if __name__ == "__main__":
    model = MovieRecommender.from_artifacts()
    print(f"Loaded {len(model.movies)} movies.")
