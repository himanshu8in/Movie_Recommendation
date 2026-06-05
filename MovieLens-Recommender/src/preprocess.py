"""
preprocess.py

Purpose:
--------
Load and preprocess MovieLens 100K dataset.

Files Used:
-----------
u.data -> Ratings
u.item -> Movie information
u.user -> User information

Returns:
--------
ratings dataframe
movies dataframe
users dataframe
"""

import pandas as pd


def load_data(data_path="../data"):

    # Load ratings dataset
    ratings = pd.read_csv(
        f"{data_path}/u.data",
        sep="\t",
        names=[
            "user_id",
            "movie_id",
            "rating",
            "timestamp"
        ]
    )

    # Load movie dataset
    movies = pd.read_csv(
        f"{data_path}/u.item",
        sep="|",
        encoding="latin-1",
        header=None
    )

    # Movie ID and Title
    movies = movies.iloc[:, :2]

    movies.columns = [
        "movie_id",
        "title"
    ]

    # Load users dataset
    users = pd.read_csv(
        f"{data_path}/u.user",
        sep="|",
        names=[
            "user_id",
            "age",
            "gender",
            "occupation",
            "zip_code"
        ]
    )

    return ratings, movies, users


def create_user_movie_matrix(ratings):

    """
    Create sparse user movie matrix

    Rows -> Users
    Columns -> Movies
    Values -> Ratings
    """

    matrix = ratings.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating"
    )

    return matrix