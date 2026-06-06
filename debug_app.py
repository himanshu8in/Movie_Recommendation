print("Step 1")
import streamlit as st

print("Step 2")
import pandas as pd

print("Step 3")
import pickle

print("Step 4")
from src.preprocess import load_data

print("Step 5")
from src.collaborative import recommend_movies

print("Step 6")
ratings, movies, users = load_data("data")

print("Ratings:", ratings.shape)
print("Movies:", movies.shape)
print("Users:", users.shape)

print("Step 7")
with open("models/svd_model.pkl", "rb") as f:
    model = pickle.load(f)

print("Step 8")
print("Everything loaded successfully")