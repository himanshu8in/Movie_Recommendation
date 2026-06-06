"""
content_based.py

Content Based Recommendation
using movie titles and genres
"""

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity


def build_similarity_matrix(movies):

    """
    Create TF-IDF representation
    of movie titles.
    """

    tfidf = TfidfVectorizer(
        stop_words="english"
    )

    tfidf_matrix = tfidf.fit_transform(
        movies["title"]
    )

    similarity_matrix = cosine_similarity(
        tfidf_matrix
    )

    return similarity_matrix


def get_similar_movies(
        movie_title,
        movies,
        similarity_matrix,
        top_n=10
):

    idx = movies[
        movies["title"] == movie_title
    ].index[0]

    scores = list(
        enumerate(
            similarity_matrix[idx]
        )
    )

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    scores = scores[1:top_n+1]

    recommendations = []

    for i in scores:

        recommendations.append(
            movies.iloc[
                i[0]
            ]["title"]
        )

    return recommendations