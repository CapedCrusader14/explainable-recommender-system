
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

RATINGS_PATH = ROOT / "data" / "ml-100k" / "u.data"
MOVIES_PATH = ROOT / "data" / "movielens_100k.csv"

RESULTS_PATH = ROOT / "data" / "evaluation_results.csv"
WEIGHTS_PATH = ROOT / "data" / "best_hybrid_weights.csv"

K = 10
RELEVANCE_THRESHOLD = 4.0
NEIGHBOR_COUNT = 30
SMOOTHING = 10.0

# Search weights in increments of 0.1.
WEIGHT_GRID = [i / 10 for i in range(11)]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    if not RATINGS_PATH.exists():
        raise FileNotFoundError(f"Missing ratings file: {RATINGS_PATH}")

    if not MOVIES_PATH.exists():
        raise FileNotFoundError(f"Missing metadata file: {MOVIES_PATH}")

    ratings = pd.read_csv(
        RATINGS_PATH,
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )

    movies = pd.read_csv(MOVIES_PATH, encoding="latin-1")

    required = {"movie_id", "title"}
    missing = required - set(movies.columns)

    if missing:
        raise ValueError(f"Missing metadata columns: {missing}")

    for column in ["user_id", "movie_id", "timestamp"]:
        ratings[column] = pd.to_numeric(
            ratings[column], errors="raise"
        ).astype(int)

    ratings["rating"] = pd.to_numeric(
        ratings["rating"], errors="raise"
    ).astype(float)

    movies["movie_id"] = pd.to_numeric(
        movies["movie_id"], errors="raise"
    ).astype(int)

    movies = movies.drop_duplicates("movie_id").copy()

    feature_columns = [
        col for col in ["genres", "directors", "actors"]
        if col in movies.columns
    ]

    if feature_columns:
        metadata_text = (
            movies[feature_columns]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
        )
    else:
        metadata_text = pd.Series("", index=movies.index)

    movies["features"] = (
        movies["title"].fillna("").astype(str)
        + " "
        + metadata_text
    )

    return ratings, movies


# ============================================================
# CHRONOLOGICAL THREE-WAY SPLIT
# ============================================================

def chronological_three_way_split(ratings):
    """
    Split each user's interactions chronologically.

    Approximate proportions:
      70% training
      10% validation
      20% test

    Each user with at least three ratings gets at least one
    training, validation, and test interaction.
    """
    train_parts = []
    validation_parts = []
    test_parts = []

    for _, group in ratings.groupby("user_id", sort=True):
        group = group.sort_values(
            ["timestamp", "movie_id"]
        ).reset_index(drop=True)

        n = len(group)

        if n < 3:
            train_parts.append(group)
            continue

        train_end = max(1, int(np.floor(n * 0.70)))
        validation_end = max(
            train_end + 1,
            int(np.floor(n * 0.80)),
        )

        validation_end = min(validation_end, n - 1)

        train_parts.append(group.iloc[:train_end])
        validation_parts.append(group.iloc[train_end:validation_end])
        test_parts.append(group.iloc[validation_end:])

    train = pd.concat(train_parts, ignore_index=True)

    validation = (
        pd.concat(validation_parts, ignore_index=True)
        if validation_parts
        else ratings.iloc[0:0].copy()
    )

    test = (
        pd.concat(test_parts, ignore_index=True)
        if test_parts
        else ratings.iloc[0:0].copy()
    )

    return train, validation, test


# ============================================================
# METADATA CONTENT MODEL
# ============================================================

def build_content_model(movies):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=50000,
    )

    matrix = vectorizer.fit_transform(
        movies["features"].fillna("")
    )

    movie_index = {
        movie_id: i
        for i, movie_id in enumerate(movies["movie_id"])
    }

    return matrix, movie_index


def content_scores_for_user(
    user_id,
    train,
    content_matrix,
    movie_index,
    candidates,
):
    user_ratings = train[train["user_id"] == user_id]

    if user_ratings.empty:
        return {movie_id: 0.0 for movie_id in candidates}

    user_mean = float(user_ratings["rating"].mean())

    indices = []
    weights = []

    for row in user_ratings.itertuples(index=False):
        idx = movie_index.get(row.movie_id)

        if idx is not None:
            indices.append(idx)
            weights.append(float(row.rating) - user_mean)

    if not indices:
        return {movie_id: 0.0 for movie_id in candidates}

    profile = content_matrix[indices].T @ np.asarray(weights)
    profile = np.asarray(profile).reshape(1, -1)

    valid_candidates = [
        movie_id for movie_id in candidates
        if movie_id in movie_index
    ]

    if not valid_candidates:
        return {movie_id: 0.0 for movie_id in candidates}

    candidate_indices = [
        movie_index[movie_id] for movie_id in valid_candidates
    ]

    similarities = cosine_similarity(
        profile,
        content_matrix[candidate_indices],
    )[0]

    scores = {
        movie_id: float(score)
        for movie_id, score in zip(valid_candidates, similarities)
    }

    return {
        movie_id: scores.get(movie_id, 0.0)
        for movie_id in candidates
    }


# ============================================================
# POPULARITY MODEL
# ============================================================

def build_popularity_model(train):
    global_mean = float(train["rating"].mean())

    stats = train.groupby("movie_id")["rating"].agg(
        ["count", "mean"]
    )

    smoothed_mean = (
        stats["count"] * stats["mean"]
        + SMOOTHING * global_mean
    ) / (stats["count"] + SMOOTHING)

    popularity = smoothed_mean + (
        0.05 * np.log1p(stats["count"])
    )

    return popularity.to_dict(), global_mean


# ============================================================
# USER-BASED COLLABORATIVE FILTERING
# ============================================================

def build_collaborative_model(train):
    rating_matrix = train.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating",
        aggfunc="mean",
    )

    user_means = rating_matrix.mean(axis=1)

    centered = rating_matrix.sub(
        user_means, axis=0
    ).fillna(0.0)

    similarities = cosine_similarity(centered)

    similarity_matrix = pd.DataFrame(
        similarities,
        index=centered.index,
        columns=centered.index,
    )

    return rating_matrix, user_means, similarity_matrix


def collaborative_scores_for_user(
    user_id,
    candidates,
    rating_matrix,
    user_means,
    similarity_matrix,
    global_mean,
):
    if user_id not in similarity_matrix.index:
        return {movie_id: global_mean for movie_id in candidates}

    similarities = similarity_matrix.loc[user_id].drop(
        labels=[user_id],
        errors="ignore",
    )

    similarities = similarities[similarities > 0]
    similarities = similarities.sort_values(
        ascending=False
    ).head(NEIGHBOR_COUNT)

    if similarities.empty:
        fallback = float(user_means.get(user_id, global_mean))
        return {movie_id: fallback for movie_id in candidates}

    neighbors = similarities.index
    neighbor_ratings = rating_matrix.loc[neighbors]
    neighbor_means = user_means.loc[neighbors]

    target_mean = float(user_means.get(user_id, global_mean))
    scores = {}

    for movie_id in candidates:
        if movie_id not in neighbor_ratings.columns:
            scores[movie_id] = target_mean
            continue

        movie_ratings = neighbor_ratings[movie_id].dropna()

        if movie_ratings.empty:
            scores[movie_id] = target_mean
            continue

        rated_neighbors = movie_ratings.index
        weights = similarities.loc[rated_neighbors].to_numpy(
            dtype=float
        )

        means = neighbor_means.loc[rated_neighbors].to_numpy(
            dtype=float
        )

        deviations = movie_ratings.to_numpy(dtype=float) - means
        denominator = np.abs(weights).sum()

        if denominator <= 1e-12:
            scores[movie_id] = target_mean
            continue

        prediction = target_mean + (
            np.dot(weights, deviations) / denominator
        )

        scores[movie_id] = float(np.clip(prediction, 1.0, 5.0))

    return scores


# ============================================================
# NORMALIZATION AND RANKING
# ============================================================

def normalize_scores(scores):
    if not scores:
        return {}

    values = np.asarray(list(scores.values()), dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Model produced non-finite scores.")

    low = float(values.min())
    high = float(values.max())

    if high - low <= 1e-12:
        return {movie_id: 0.0 for movie_id in scores}

    return {
        movie_id: float((score - low) / (high - low))
        for movie_id, score in scores.items()
    }


def rank_movies(scores):
    # Stable, deterministic tie-breaking.
    return sorted(
        scores,
        key=lambda movie_id: (-scores[movie_id], movie_id),
    )[:K]


# ============================================================
# RELEVANCE AND METRICS
# ============================================================

def make_relevant_sets(evaluation_data, history):
    relevant_by_user = {}

    for user_id, group in evaluation_data.groupby("user_id"):
        seen = history.get(user_id, set())

        relevant = set(
            group.loc[
                (group["rating"] >= RELEVANCE_THRESHOLD)
                & (~group["movie_id"].isin(seen)),
                "movie_id",
            ]
        )

        if relevant:
            relevant_by_user[int(user_id)] = relevant

    return relevant_by_user


def precision_at_k(recommended, relevant):
    return sum(
        movie_id in relevant for movie_id in recommended[:K]
    ) / K


def recall_at_k(recommended, relevant):
    if not relevant:
        return 0.0

    hits = sum(
        movie_id in relevant for movie_id in recommended[:K]
    )

    return hits / len(relevant)


def ndcg_at_k(recommended, relevant):
    if not relevant:
        return 0.0

    dcg = sum(
        1.0 / np.log2(rank + 1)
        for rank, movie_id in enumerate(recommended[:K], start=1)
        if movie_id in relevant
    )

    ideal_count = min(len(relevant), K)

    idcg = sum(
        1.0 / np.log2(rank + 1)
        for rank in range(1, ideal_count + 1)
    )

    return dcg / idcg if idcg else 0.0


def average_precision_at_k(recommended, relevant):
    if not relevant:
        return 0.0

    hits = 0
    precision_sum = 0.0

    for rank, movie_id in enumerate(recommended[:K], start=1):
        if movie_id in relevant:
            hits += 1
            precision_sum += hits / rank

    return precision_sum / min(len(relevant), K)


def reciprocal_rank(recommended, relevant):
    for rank, movie_id in enumerate(recommended, start=1):
        if movie_id in relevant:
            return 1.0 / rank

    return 0.0


def evaluate_recommendations(recommendations, relevant_by_user):
    metric_values = {
        "Precision@K": [],
        "Recall@K": [],
        "NDCG@K": [],
        "MAP@K": [],
        "MRR": [],
    }

    for user_id, relevant in relevant_by_user.items():
        recommended = recommendations.get(user_id, [])

        metric_values["Precision@K"].append(
            precision_at_k(recommended, relevant)
        )
        metric_values["Recall@K"].append(
            recall_at_k(recommended, relevant)
        )
        metric_values["NDCG@K"].append(
            ndcg_at_k(recommended, relevant)
        )
        metric_values["MAP@K"].append(
            average_precision_at_k(recommended, relevant)
        )
        metric_values["MRR"].append(
            reciprocal_rank(recommended, relevant)
        )

    return {
        metric: float(np.mean(values)) if values else 0.0
        for metric, values in metric_values.items()
    }


# ============================================================
# FIT MODELS AND GENERATE COMPONENT SCORES
# ============================================================

def generate_component_scores(
    fit_ratings,
    evaluation_users,
    movies,
    catalog_ids,
    history,
):
    """
    Fit components on fit_ratings, then score each user's unseen
    candidate movies for the specified evaluation users.
    """
    content_matrix, movie_index = build_content_model(movies)

    popularity, global_mean = build_popularity_model(fit_ratings)

    rating_matrix, user_means, similarity_matrix = (
        build_collaborative_model(fit_ratings)
    )

    output = {}

    for user_id in evaluation_users:
        seen = history.get(user_id, set())

        candidates = [
            movie_id for movie_id in catalog_ids
            if movie_id not in seen
        ]

        content_raw = content_scores_for_user(
            user_id,
            fit_ratings,
            content_matrix,
            movie_index,
            candidates,
        )

        collaborative_raw = collaborative_scores_for_user(
            user_id,
            candidates,
            rating_matrix,
            user_means,
            similarity_matrix,
            global_mean,
        )

        popularity_raw = {
            movie_id: float(popularity.get(movie_id, global_mean))
            for movie_id in candidates
        }

        output[user_id] = {
            "candidates": candidates,
            "content": normalize_scores(content_raw),
            "collaborative": normalize_scores(collaborative_raw),
            "popularity": normalize_scores(popularity_raw),
        }

    return output


def build_hybrid_scores(component_scores, weights):
    content_weight, collab_weight, popularity_weight = weights

    hybrid = {}

    for movie_id in component_scores["candidates"]:
        hybrid[movie_id] = (
            content_weight * component_scores["content"].get(
                movie_id, 0.0
            )
            + collab_weight * component_scores["collaborative"].get(
                movie_id, 0.0
            )
            + popularity_weight * component_scores["popularity"].get(
                movie_id, 0.0
            )
        )

    return hybrid


def recommendations_for_components(component_scores, weights):
    """
    Return rankings for all component models and the hybrid.
    """
    recommendations = {
        "Popularity Baseline": {},
        "User-Profile TF-IDF": {},
        "User-Based Collaborative Filtering": {},
        "Hybrid": {},
    }

    for user_id, components in component_scores.items():
        recommendations["Popularity Baseline"][user_id] = rank_movies(
            components["popularity"]
        )

        recommendations["User-Profile TF-IDF"][user_id] = rank_movies(
            components["content"]
        )

        recommendations["User-Based Collaborative Filtering"][user_id] = (
            rank_movies(components["collaborative"])
        )

        recommendations["Hybrid"][user_id] = rank_movies(
            build_hybrid_scores(components, weights)
        )

    return recommendations


# ============================================================
# GRID SEARCH ON VALIDATION ONLY
# ============================================================

def tune_weights(validation_components, validation_relevant):
    """
    Exhaustively search all weights in increments of 0.1 that sum
    to one. Select the weights with the highest validation NDCG@10.

    Ties are broken by MAP@10, then Precision@10.
    """
    best_weights = None
    best_metrics = None
    best_key = None
    search_rows = []

    for content_weight in WEIGHT_GRID:
        for collab_weight in WEIGHT_GRID:
            popularity_weight = round(
                1.0 - content_weight - collab_weight, 1
            )

            if popularity_weight < -1e-9:
                continue

            if not any(
                abs(popularity_weight - value) < 1e-9
                for value in WEIGHT_GRID
            ):
                continue

            weights = (
                round(content_weight, 1),
                round(collab_weight, 1),
                round(max(0.0, popularity_weight), 1),
            )

            hybrid_recommendations = {}

            for user_id, components in validation_components.items():
                scores = build_hybrid_scores(components, weights)
                hybrid_recommendations[user_id] = rank_movies(scores)

            metrics = evaluate_recommendations(
                hybrid_recommendations,
                validation_relevant,
            )

            search_rows.append({
                "content_weight": weights[0],
                "collaborative_weight": weights[1],
                "popularity_weight": weights[2],
                **metrics,
            })

            key = (
                metrics["NDCG@K"],
                metrics["MAP@K"],
                metrics["Precision@K"],
            )

            if best_key is None or key > best_key:
                best_key = key
                best_weights = weights
                best_metrics = metrics

    search_df = pd.DataFrame(search_rows).sort_values(
        ["NDCG@K", "MAP@K", "Precision@K"],
        ascending=False,
    )

    return best_weights, best_metrics, search_df


# ============================================================
# MAIN
# ============================================================

def main():
    ratings, movies = load_data()

    train, validation, test = chronological_three_way_split(ratings)

    metadata_ids = set(movies["movie_id"])
    catalog_ids = sorted(
        metadata_ids & set(ratings["movie_id"])
    )

    # --------------------------------------------------------
    # VALIDATION PHASE
    # --------------------------------------------------------

    train_history = (
        train.groupby("user_id")["movie_id"].apply(set).to_dict()
    )

    validation_relevant = make_relevant_sets(
        validation,
        train_history,
    )

    validation_users = sorted(validation_relevant)

    if not validation_users:
        raise RuntimeError(
            "No users have relevant validation movies. "
            "Check the dataset and relevance threshold."
        )

    print("Fitting models on training data for validation tuning...")

    validation_components = generate_component_scores(
        fit_ratings=train,
        evaluation_users=validation_users,
        movies=movies,
        catalog_ids=catalog_ids,
        history=train_history,
    )

    best_weights, best_validation_metrics, search_df = tune_weights(
        validation_components,
        validation_relevant,
    )

    print("\nVALIDATION RESULTS")
    print(f"Users evaluated: {len(validation_users)}")
    print(f"Best content weight: {best_weights[0]:.1f}")
    print(f"Best collaborative weight: {best_weights[1]:.1f}")
    print(f"Best popularity weight: {best_weights[2]:.1f}")
    print("\nBest validation metrics:")

    for metric, value in best_validation_metrics.items():
        print(f"{metric}: {value:.4f}")

    search_df.to_csv(
        ROOT / "data" / "hybrid_weight_search.csv",
        index=False,
    )

    # --------------------------------------------------------
    # FINAL TEST PHASE
    # --------------------------------------------------------

    # Refit on train + validation after the weights have been selected.
    final_train = pd.concat(
        [train, validation],
        ignore_index=True,
    )

    final_history = (
        final_train.groupby("user_id")["movie_id"]
        .apply(set)
        .to_dict()
    )

    test_relevant = make_relevant_sets(
        test,
        final_history,
    )

    test_users = sorted(test_relevant)

    if not test_users:
        raise RuntimeError(
            "No users have relevant test movies. "
            "Check the split and relevance threshold."
        )

    print("\nFitting final models on training + validation data...")

    test_components = generate_component_scores(
        fit_ratings=final_train,
        evaluation_users=test_users,
        movies=movies,
        catalog_ids=catalog_ids,
        history=final_history,
    )

    test_recommendations = recommendations_for_components(
        test_components,
        best_weights,
    )

    test_results = {}

    for model_name, recommendations in test_recommendations.items():
        test_results[model_name] = evaluate_recommendations(
            recommendations,
            test_relevant,
        )

    results_df = pd.DataFrame(test_results).round(4)

    # Save test metrics only; validation metrics are printed separately.
    results_df.to_csv(
        RESULTS_PATH,
        index=True,
        index_label="Metric",
    )

    weights_df = pd.DataFrame([{
        "content_weight": best_weights[0],
        "collaborative_weight": best_weights[1],
        "popularity_weight": best_weights[2],
        "validation_ndcg_at_10": best_validation_metrics["NDCG@K"],
        "validation_map_at_10": best_validation_metrics["MAP@K"],
        "validation_precision_at_10": best_validation_metrics["Precision@K"],
    }])

    weights_df.to_csv(WEIGHTS_PATH, index=False)

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print("\nDATA DIAGNOSTICS")
    print(f"Total ratings: {len(ratings)}")
    print(f"Training ratings: {len(train)}")
    print(f"Validation ratings: {len(validation)}")
    print(f"Final fit ratings (train + validation): {len(final_train)}")
    print(f"Test ratings: {len(test)}")
    print(f"Users: {ratings['user_id'].nunique()}")
    print(f"Movies in ratings: {ratings['movie_id'].nunique()}")
    print(f"Movies in metadata: {len(movies)}")
    print(f"Candidate movies: {len(catalog_ids)}")
    print(f"Validation users evaluated: {len(validation_users)}")
    print(f"Test users evaluated: {len(test_users)}")
    print(f"Relevance threshold: rating >= {RELEVANCE_THRESHOLD}")
    print(f"K: {K}")
    print("Split: chronological within each user")

    print("\nSELECTED HYBRID WEIGHTS")
    print(f"Content: {best_weights[0]:.1f}")
    print(f"Collaborative: {best_weights[1]:.1f}")
    print(f"Popularity: {best_weights[2]:.1f}")

    print("\nFINAL TEST RESULTS")
    print(results_df.to_string())

    print(f"\nSaved test results to: {RESULTS_PATH}")
    print(f"Saved selected weights to: {WEIGHTS_PATH}")
    print(
        "Saved all validation weight combinations to: "
        f"{ROOT / 'data' / 'hybrid_weight_search.csv'}"
    )

    # --------------------------------------------------------
    # MODEL DIVERSITY DIAGNOSTIC
    # --------------------------------------------------------

    content_lists = test_recommendations["User-Profile TF-IDF"]
    collab_lists = test_recommendations[
        "User-Based Collaborative Filtering"
    ]

    identical_count = sum(
        content_lists[user_id] == collab_lists[user_id]
        for user_id in test_users
    )

    overlaps = [
        len(
            set(content_lists[user_id])
            & set(collab_lists[user_id])
        ) / K
        for user_id in test_users
    ]

    print("\nCONTENT VS COLLABORATIVE DIAGNOSTIC")
    print(
        f"Identical top-{K} lists: "
        f"{identical_count}/{len(test_users)}"
    )
    print(
        f"Average top-{K} overlap: {np.mean(overlaps):.2%}"
    )

    # --------------------------------------------------------
    # EXAMPLE TEST USER
    # --------------------------------------------------------

    example_user = test_users[0]

    print(f"\nEXAMPLE TEST USER: {example_user}")
    print("Relevant held-out movies:", sorted(test_relevant[example_user]))

    for model_name, recommendations in test_recommendations.items():
        print(f"\n{model_name}:")
        print(recommendations[example_user])


if __name__ == "__main__":
    main()
