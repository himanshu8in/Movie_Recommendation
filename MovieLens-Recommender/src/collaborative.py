"""
collaborative.py

Collaborative Filtering using SVD
"""

from surprise import Dataset
from surprise import Reader
from surprise import SVD

import numpy as np


def prepare_surprise_dataset(ratings):

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

    return data


def train_svd(trainset):

    """
    Train SVD model.

    n_factors:
        Number of latent features

    n_epochs:
        Number of training iterations
    """

    model = SVD(
        n_factors=100,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.02
    )

    model.fit(trainset)

    return model


def recommend_movies(
        user_id,
        ratings,
        model,
        top_n=10
):

    all_movies = ratings["movie_id"].unique()

    watched_movies = ratings[
        ratings["user_id"] == user_id
    ]["movie_id"].values

    unseen_movies = np.setdiff1d(
        all_movies,
        watched_movies
    )

    predictions = []

    for movie in unseen_movies:

        pred = model.predict(
            user_id,
            movie
        )

        predictions.append(
            (
                movie,
                pred.est
            )
        )

    predictions.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return predictions[:top_n]