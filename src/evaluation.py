"""
evaluation.py

Model evaluation metrics
"""

from surprise import accuracy


def calculate_rmse(
        model,
        testset
):

    predictions = model.test(
        testset
    )

    rmse = accuracy.rmse(
        predictions,
        verbose=True
    )

    return rmse


def calculate_mae(
        model,
        testset
):

    predictions = model.test(
        testset
    )

    mae = accuracy.mae(
        predictions,
        verbose=True
    )

    return mae