from pathlib import Path
import pickle

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
FEATURE_STORE_DIR = PROJECT_DIR / "Feature_store"
MODEL_DIR = PROJECT_DIR / "models"
USER_ID = 45


def ingest_data(data_dir: Path = PROJECT_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings_path = data_dir / "ratings.csv"
    movies_path = data_dir / "movies.csv"
    missing_files = [path.name for path in (ratings_path, movies_path) if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"Required dataset file(s) not found in {data_dir}: {', '.join(missing_files)}"
        )

    return pd.read_csv(ratings_path), pd.read_csv(movies_path)


def validate_data(ratings: pd.DataFrame, movies: pd.DataFrame) -> None:
    required_rating_columns = {"userId", "movieId", "rating", "timestamp"}
    required_movie_columns = {"movieId", "title", "genres"}

    if not required_rating_columns.issubset(ratings.columns):
        raise ValueError("Ratings data is missing required columns.")
    if not required_movie_columns.issubset(movies.columns):
        raise ValueError("Movies data is missing required columns.")
    if ratings[list(required_rating_columns)].isnull().any().any():
        raise ValueError("Ratings data contains missing values in required columns.")
    if movies[list(required_movie_columns)].isnull().any().any():
        raise ValueError("Movies data contains missing values in required columns.")
    if not ratings["rating"].between(0.5, 5.0).all():
        raise ValueError("Ratings must be between 0.5 and 5.0.")

    unknown_movie_ids = set(ratings["movieId"]) - set(movies["movieId"])
    if unknown_movie_ids:
        raise ValueError("Ratings data contains movie IDs missing from movies.csv.")


def prepare_data(
    ratings: pd.DataFrame, movies: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings_clean = ratings.drop_duplicates(
        subset=["userId", "movieId"], keep="first"
    ).copy()
    ratings_clean["rating_datetime"] = pd.to_datetime(
        ratings_clean["timestamp"], unit="s"
    )
    movies_clean = movies.drop_duplicates(subset=["movieId"], keep="first").copy()
    return ratings_clean, movies_clean


def transform_data(
    ratings: pd.DataFrame, movies: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    recommendation_data = ratings.merge(movies, on="movieId", how="inner")

    movie_stats = (
        recommendation_data.groupby(["movieId", "title"])
        .agg(
            avg_movie_rating=("rating", "mean"),
            movie_rating_count=("rating", "count"),
        )
        .reset_index()
    )
    user_stats = (
        recommendation_data.groupby("userId")
        .agg(
            avg_user_rating=("rating", "mean"),
            user_rating_count=("rating", "count"),
        )
        .reset_index()
    )

    feature_data = recommendation_data.merge(
        movie_stats, on=["movieId", "title"], how="left"
    ).merge(user_stats, on="userId", how="left")
    return movie_stats, user_stats, feature_data


def manage_features(
    movie_stats: pd.DataFrame,
    user_stats: pd.DataFrame,
    feature_data: pd.DataFrame,
    feature_store_dir: Path = FEATURE_STORE_DIR,
) -> None:
    feature_store_dir.mkdir(parents=True, exist_ok=True)
    movie_stats.to_csv(feature_store_dir / "movie_stats.csv", index=False)
    user_stats.to_csv(feature_store_dir / "user_features.csv", index=False)
    feature_data.to_csv(
        feature_store_dir / "recommendation_features.csv", index=False
    )


def train_model(
    feature_data: pd.DataFrame, model_dir: Path = MODEL_DIR
) -> tuple[pd.DataFrame, pd.DataFrame]:
    user_movie_matrix = feature_data.pivot_table(
        index="userId", columns="movieId", values="rating"
    )
    ratings = user_movie_matrix.fillna(0).to_numpy(dtype=float)
    dot_products = ratings @ ratings.T
    norms = np.linalg.norm(ratings, axis=1)
    norm_products = np.outer(norms, norms)
    similarities = np.divide(
        dot_products,
        norm_products,
        out=np.zeros_like(dot_products),
        where=norm_products != 0,
    )
    user_similarity = pd.DataFrame(
        similarities,
        index=user_movie_matrix.index,
        columns=user_movie_matrix.index,
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "user_similarity_model.pkl"
    with model_path.open("wb") as model_file:
        pickle.dump(user_similarity, model_file)

    return user_movie_matrix, user_similarity


def generate_recommendations(
    model: tuple[pd.DataFrame, pd.DataFrame],
    movies: pd.DataFrame,
    user_id: int = USER_ID,
    n_recommendations: int = 5,
    n_similar_users: int = 20,
) -> pd.DataFrame:
    user_movie_matrix, user_similarity = model
    if user_id not in user_movie_matrix.index:
        raise ValueError(f"User {user_id} does not exist in the ratings data.")

    similar_users = (
        user_similarity[user_id].drop(index=user_id).nlargest(n_similar_users)
    )
    neighbor_ratings = user_movie_matrix.loc[similar_users.index]
    weights = similar_users.to_numpy(dtype=float)
    known_ratings = neighbor_ratings.notna().to_numpy(dtype=float)
    weighted_scores = neighbor_ratings.fillna(0).to_numpy(dtype=float).T @ weights
    similarity_totals = known_ratings.T @ weights

    target_ratings = user_movie_matrix.loc[user_id]
    eligible = target_ratings.isna().to_numpy() & (similarity_totals > 0)
    movie_ids = user_movie_matrix.columns.to_numpy()
    eligible_positions = np.flatnonzero(eligible)
    if eligible_positions.size == 0:
        raise ValueError(f"No unrated recommendations are available for user {user_id}.")

    predicted_ratings = weighted_scores[eligible_positions] / similarity_totals[
        eligible_positions
    ]
    top_positions = np.argsort(predicted_ratings)[::-1][:n_recommendations]
    top_recommendations = pd.DataFrame(
        {
            "movieId": movie_ids[eligible_positions[top_positions]],
            "predicted_rating": predicted_ratings[top_positions],
        }
    )
    recommendations = top_recommendations.merge(
        movies[["movieId", "title", "genres"]],
        on="movieId",
        how="left",
    )
    return recommendations[
        ["movieId", "title", "genres", "predicted_rating"]
    ]


def main() -> None:
    ratings, movies = ingest_data()
    print("[1/7] Data ingestion completed")
    print(f"Ratings: {len(ratings)}")

    validate_data(ratings, movies)
    print("\n[2/7] Data validation passed")

    ratings_clean, movies_clean = prepare_data(ratings, movies)
    print("\n[3/7] Data preparation completed")

    movie_stats, user_stats, feature_data = transform_data(ratings_clean, movies_clean)
    print("\n[4/7] Data transformation completed")

    manage_features(movie_stats, user_stats, feature_data)
    print("\n[5/7] Features stored successfully")

    model = train_model(feature_data)
    print("\n[6/7] Recommendation model trained")
    print("Model saved successfully")

    recommendations = generate_recommendations(model, movies_clean)
    print("\n[7/7] Recommendations generated")
    print("\nTop 5 Recommendations:")
    print(
        recommendations.to_string(
            index=False, formatters={"predicted_rating": "{:.2f}".format}
        )
    )


if __name__ == "__main__":
    main()
