"""
hybrid.py

Hybrid Recommendation System

Final Score =
0.7 * SVD
+
0.3 * Content Similarity
"""

import pandas as pd


def hybrid_recommendation(
        user_id,
        movie_title,
        ratings,
        movies,
        model,
        similarity_matrix,
        top_n=10
):

    idx = movies[
        movies["title"] == movie_title
    ].index[0]

    similarity_scores = list(
        enumerate(
            similarity_matrix[idx]
        )
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for movie_index, content_score in similarity_scores[1:100]:

        movie_id = movies.iloc[
            movie_index
        ]["movie_id"]

        svd_score = model.predict(
            user_id,
            movie_id
        ).est

        hybrid_score = (
            0.7 * svd_score
            +
            0.3 * content_score
        )

        recommendations.append(
            (
                movie_id,
                hybrid_score
            )
        )

    recommendations = sorted(
        recommendations,
        key=lambda x: x[1],
        reverse=True
    )

    return recommendations[:top_n]