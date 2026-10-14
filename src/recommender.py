
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Recommender:
    def __init__(self, csv_path="data/movielens_100k.csv"):
        self.movies = pd.read_csv(csv_path)

        # Handle missing metadata
        for column in ["directors", "actors", "genres"]:
            self.movies[column] = (
                self.movies[column].fillna("").astype(str)
            )

        # Combine metadata into one searchable description
        self.movies["features"] = (
            self.movies["genres"] + " " +
            self.movies["directors"] + " " +
            self.movies["actors"]
        )

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            token_pattern=r"(?u)\b\w+\b"
        )
        self.matrix = self.vectorizer.fit_transform(
            self.movies["features"]
        )
        self.similarity = cosine_similarity(self.matrix)

    def recommend(self, movie_id, top_n=5):
        matches = self.movies.index[
            self.movies["movie_id"] == movie_id
        ].tolist()

        if not matches:
            raise ValueError(f"Movie ID {movie_id} was not found.")

        index = matches[0]
        scores = self.similarity[index]
        ranked = scores.argsort()[::-1]
        ranked = [i for i in ranked if i != index][:top_n]

        results = self.movies.iloc[ranked][
            ["movie_id", "title", "year", "genres"]
        ].copy()
        results["similarity"] = [
            round(float(scores[i]), 3) for i in ranked
        ]

        return results.reset_index(drop=True)
