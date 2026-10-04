"""Command-line entry point for generating recommendation artifacts."""

from recommender import build_artifacts


if __name__ == "__main__":
    movies, vectors = build_artifacts()
    print(
        f"Built artifacts for {len(movies)} movies "
        f"({vectors.shape[0]} x {vectors.shape[1]} sparse feature matrix)."
    )