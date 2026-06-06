import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('ggplot')

ratings = pd.read_csv(
    "MovieLens-Recommender/data/u.data",
    sep = "\t",
    names = ["user_id","movie_id","rating","timestamp"]
)

movies = pd.read_csv(
    "MovieLens-Recommender/data/u.item",
    sep = "|",
    encoding="latin-1",
    usecols=[0,1],
    names=["movie_id","title"]
)

users = pd.read_csv(
    "MovieLens-Recommender/data/u.user",
    sep="|",
    names=["user_id","age","gender","occupation","zip_code"]
)

# print("Ratings Shape :",ratings.shape)
# print("Movies Shape:",movies.shape)
# print("Users shape :",users.shape)

# print(ratings.head())
# print(movies.head())
# print(users.head())

# print(ratings.isnull().sum())
# print(movies.isnull().sum())
# print(users.isnull().sum())

# print(ratings.describe())
# print(movies.describe())
# print(users.describe())

# print("Number of Users:", ratings['user_id'].nunique())
# print("Number of Movies:", ratings['movie_id'].nunique())
# print("Total Ratings:", len(ratings))

#Rating Distribution
plt.figure(figsize=(8,5))
sns.countplot(
    x = 'rating',
    data=ratings
)
plt.title("Rating Distribution")
# plt.show()

#Ratings per user
user_ratings = ratings.groupby(
    'user_id'
)['rating'].count()
#print(user_ratings.describe())

#Rating per user distribution
plt.figure(figsize=(10,5))

sns.histplot(
    user_ratings,
    bins=30
)

plt.title("Ratings Per User")
plt.show()

#Rating per Movie
movie_ratings = ratings.groupby(
    'movie_id'
)['rating'].count()

movie_ratings.describe()

plt.figure(figsize=(10,5))

sns.histplot(
    movie_ratings,
    bins=30
)

plt.title("Ratings Per Movie")
plt.show()

movie_stats = ratings.groupby(
    'movie_id'
)['rating'].count().reset_index()

movie_stats.columns = [
    'movie_id',
    'rating_count'
]

top_movies = movie_stats.merge(
    movies,
    on='movie_id'
)

top_movies.sort_values(
    by='rating_count',
    ascending=False
).head(10)

top10 = top_movies.sort_values(
    by='rating_count',
    ascending=False
).head(10)

plt.figure(figsize=(12,6))

sns.barplot(
    x='rating_count',
    y='title',
    data=top10
)

plt.title("Top 10 Most Rated Movies")
plt.show()

avg_rating = ratings.groupby(
    'movie_id'
)['rating'].mean().reset_index()

avg_rating.columns = [
    'movie_id',
    'avg_rating'
]

avg_rating = avg_rating.merge(
    movies,
    on='movie_id'
)

avg_rating.sort_values(
    by='avg_rating',
    ascending=False
).head(10)

rating_summary = ratings.groupby(
    'movie_id'
).agg({
    'rating':['mean','count']
})

rating_summary.columns = [
    'avg_rating',
    'rating_count'
]

rating_summary.reset_index(
    inplace=True
)

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=rating_summary,
    x='rating_count',
    y='avg_rating'
)

plt.title(
    "Popularity vs Average Rating"
)

plt.show()

sns.countplot(
    x='gender',
    data=users
)

plt.title("Gender Distribution")
plt.show()

sns.countplot(
    x='gender',
    data=users
)

plt.title("Gender Distribution")
plt.show()

plt.figure(figsize=(12,6))

users['occupation'].value_counts().plot(
    kind='bar'
)

plt.title("Occupation Distribution")
plt.show()

n_users = ratings.user_id.nunique()

n_movies = ratings.movie_id.nunique()

possible_ratings = n_users * n_movies

actual_ratings = len(ratings)

sparsity = (
    1 - actual_ratings/possible_ratings
) * 100

print(
    f"Sparsity: {sparsity:.2f}%"
)
user_movie_matrix = ratings.pivot_table(
    index='user_id',
    columns='movie_id',
    values='rating'
)

print(user_movie_matrix.shape)