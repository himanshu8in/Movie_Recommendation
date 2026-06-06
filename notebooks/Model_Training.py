"""
====================================================
MovieLens 100K - Model Training Script
====================================================

This script performs:

1. Load MovieLens 100K ratings dataset
2. Create Surprise Dataset
3. Train-Test Split
4. Train SVD Collaborative Filtering Model
5. Evaluate Model Performance
6. Generate Top-N Recommendations
7. Save Trained Model

Author: Himanshu Singh
====================================================
"""

# ==========================
# Import Libraries
# ==========================

import os
import pickle
import pandas as pd
import numpy as np

from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import accuracy

from surprise.model_selection import (
    train_test_split,
    cross_validate
)

# ==========================
# Load Dataset
# ==========================

print("=" * 50)
print("Loading Dataset...")
print("=" * 50)

ratings = pd.read_csv(
    "MovieLens-Recommender/data/u.data",
    sep="\t",
    names=[
        "user_id",
        "movie_id",
        "rating",
        "timestamp"
    ]
)

movies = pd.read_csv(
    "MovieLens-Recommender/data/u.item",
    sep="|",
    encoding="latin-1",
    header=None,
    usecols=[0, 1],
    names=[
        "movie_id",
        "title"
    ]
)

print("\nRatings Shape:", ratings.shape)
print("Movies Shape:", movies.shape)

# ==========================
# Dataset Statistics
# ==========================

print("\nDataset Statistics")
print("-" * 30)

print(
    "Number of Users:",
    ratings["user_id"].nunique()
)

print(
    "Number of Movies:",
    ratings["movie_id"].nunique()
)

print(
    "Total Ratings:",
    len(ratings)
)

# ==========================
# Create Surprise Dataset
# ==========================

print("\nCreating Surprise Dataset...")

reader = Reader(
    rating_scale=(1, 5)
)

data = Dataset.load_from_df(
    ratings[
        [
            "user_id",
            "movie_id",
            "rating"
        ]
    ],
    reader
)

# ==========================
# Train Test Split
# ==========================

print("\nSplitting Dataset...")

trainset, testset = train_test_split(
    data,
    test_size=0.20,
    random_state=42
)

print(
    f"Training Samples: {trainset.n_ratings}"
)

print(
    f"Testing Samples: {len(testset)}"
)

# ==========================
# Train SVD Model
# ==========================

print("\nTraining SVD Model...")

"""
SVD Hyperparameters

n_factors:
    Number of latent features

n_epochs:
    Number of iterations

lr_all:
    Learning Rate

reg_all:
    Regularization
"""

model = SVD(
    n_factors=100,
    n_epochs=20,
    lr_all=0.005,
    reg_all=0.02,
    random_state=42
)

model.fit(trainset)

print("Training Completed!")

# ==========================
# Evaluate Model
# ==========================

print("\nEvaluating Model...")

predictions = model.test(testset)

rmse = accuracy.rmse(
    predictions,
    verbose=True
)

mae = accuracy.mae(
    predictions,
    verbose=True
)

print("\nModel Performance")
print("-" * 30)

print(
    f"RMSE : {rmse:.4f}"
)

print(
    f"MAE  : {mae:.4f}"
)

# ==========================
# Cross Validation
# ==========================

print("\nPerforming 5-Fold Cross Validation...")

cv_results = cross_validate(
    SVD(),
    data,
    measures=["RMSE", "MAE"],
    cv=5,
    verbose=True
)

print("\nCross Validation Completed")

# ==========================
# Recommendation Function
# ==========================

def recommend_movies(
        user_id,
        ratings_df,
        movies_df,
        trained_model,
        top_n=10
):
    """
    Generate Top-N Movie Recommendations

    Parameters
    ----------
    user_id : int

    ratings_df : DataFrame

    movies_df : DataFrame

    trained_model : SVD Model

    top_n : int

    Returns
    -------
    Recommended Movies
    """

    # All available movies
    all_movies = ratings_df[
        "movie_id"
    ].unique()

    # Movies already watched
    watched_movies = ratings_df[
        ratings_df["user_id"] == user_id
    ]["movie_id"].unique()

    # Movies not watched
    unseen_movies = np.setdiff1d(
        all_movies,
        watched_movies
    )

    predictions = []

    # Predict rating for each unseen movie
    for movie_id in unseen_movies:

        estimated_rating = trained_model.predict(
            uid=user_id,
            iid=movie_id
        ).est

        predictions.append(
            (
                movie_id,
                estimated_rating
            )
        )

    # Sort by predicted rating
    predictions.sort(
        key=lambda x: x[1],
        reverse=True
    )

    # Top N movies
    top_movies = predictions[:top_n]

    recommendation_df = pd.DataFrame(
        top_movies,
        columns=[
            "movie_id",
            "predicted_rating"
        ]
    )

    recommendation_df = recommendation_df.merge(
        movies_df,
        on="movie_id"
    )

    return recommendation_df[
        [
            "movie_id",
            "title",
            "predicted_rating"
        ]
    ]


# ==========================
# Example Recommendation
# ==========================

print("\nGenerating Recommendations...")

sample_user = 10

recommendations = recommend_movies(
    user_id=sample_user,
    ratings_df=ratings,
    movies_df=movies,
    trained_model=model,
    top_n=10
)

print(
    f"\nTop Recommendations For User {sample_user}"
)

print("-" * 60)

print(
    recommendations.to_string(
        index=False
    )
)

# ==========================
# Save Model
# ==========================

print("\nSaving Model...")

os.makedirs(
    "models",
    exist_ok=True
)

model_path = "Movie:models/svd_model.pkl"

with open(
        model_path,
        "wb"
) as file:

    pickle.dump(
        model,
        file
    )

print(
    f"Model Saved Successfully -> {model_path}"
)

# ==========================
# Load Model Test
# ==========================

print("\nTesting Saved Model...")

with open(
        model_path,
        "rb"
) as file:

    loaded_model = pickle.load(
        file
    )

prediction = loaded_model.predict(
    uid=1,
    iid=50
)

print(
    f"\nPredicted Rating: {prediction.est:.2f}"
)

# ==========================
# End
# ==========================

print("\n" + "=" * 50)
print("Training Pipeline Completed Successfully!")
print("=" * 50)