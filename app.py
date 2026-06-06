import streamlit as st
import pandas as pd
import pickle
import requests
from pathlib import Path

from src.preprocess import load_data
from src.collaborative import recommend_movies

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="CineSuggest AI",
    page_icon="🎬",
    layout="wide"
)

TMDB_API_KEY = "6d697e7c52b2313dfccb15cc66b6fc79"

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.block-container {
    padding-top: 2rem;
}

h1 {
    text-align:center;
    color:#FF4B4B;
}

.movie-card {
    background-color:#1A1D24;
    padding:15px;
    border-radius:15px;
    text-align:center;
    margin-bottom:20px;
    box-shadow:0px 4px 10px rgba(0,0,0,0.4);
}

.movie-title {
    font-size:18px;
    font-weight:bold;
    color:white;
}

.rating {
    color:#FFD700;
    font-size:16px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_dataset():
    return load_data("data")

ratings, movies, users = load_dataset()

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    model_path = Path("models") / "svd_model.pkl"

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model

model = load_model()

# =====================================================
# TMDB POSTER FUNCTION
# =====================================================

@st.cache_data
def fetch_movie_poster(movie_name):

    url = "https://api.themoviedb.org/3/search/movie"

    params = {
        "api_key": TMDB_API_KEY,
        "query": movie_name
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=5
        )

        data = response.json()

        if len(data["results"]) > 0:

            poster_path = data["results"][0]["poster_path"]

            if poster_path:

                return (
                    "https://image.tmdb.org/t/p/w500"
                    + poster_path
                )

    except:
        pass

    return None

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🎬 CineSuggest AI")

page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Recommendations",
        "Analytics",
        "About Model"
    ]
)

# =====================================================
# HOME PAGE
# =====================================================

if page == "Home":

    st.title("🎬 CineSuggest AI")

    st.markdown(
        "### Hybrid Movie Recommendation System"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "👤 Users",
            ratings["user_id"].nunique()
        )

    with col2:
        st.metric(
            "🎥 Movies",
            ratings["movie_id"].nunique()
        )

    with col3:
        st.metric(
            "⭐ Ratings",
            len(ratings)
        )

    st.markdown("---")

    st.subheader("Project Overview")

    st.write("""
    This recommendation system uses
    Collaborative Filtering (SVD)
    to generate personalized
    movie recommendations.
    """)

# =====================================================
# RECOMMENDATIONS PAGE
# =====================================================

elif page == "Recommendations":

    st.title("🎯 Personalized Recommendations")

    user_id = st.number_input(
        "Enter User ID",
        min_value=1,
        max_value=943,
        value=10
    )

    if st.button(
        "Generate Recommendations"
    ):

        recs = recommend_movies(
            user_id,
            ratings,
            model,
            top_n=12
        )

        rec_df = pd.DataFrame(
            recs,
            columns=[
                "movie_id",
                "predicted_rating"
            ]
        )

        rec_df = rec_df.merge(
            movies,
            on="movie_id"
        )

        st.success(
            f"Top recommendations for User {user_id}"
        )

        cols = st.columns(4)

        for idx, row in rec_df.iterrows():

            poster = fetch_movie_poster(
                row["title"]
            )

            with cols[idx % 4]:

                if poster:
                    st.image(
                        poster,
                        use_container_width=True
                    )

                st.markdown(
                    f"""
                    <div class="movie-card">
                        <div class="movie-title">
                            {row['title']}
                        </div>

                        <div class="rating">
                            ⭐ {row['predicted_rating']:.2f}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# =====================================================
# ANALYTICS PAGE
# =====================================================

elif page == "Analytics":

    import matplotlib.pyplot as plt

    st.title("📊 Analytics Dashboard")

    st.subheader(
        "Rating Distribution"
    )

    fig, ax = plt.subplots(
        figsize=(8,4)
    )

    ratings["rating"].value_counts() \
        .sort_index() \
        .plot(
            kind="bar",
            ax=ax
        )

    st.pyplot(fig)

    st.subheader(
        "Top Rated Movies"
    )

    top_movies = (
        ratings.groupby("movie_id")
        ["rating"]
        .mean()
        .sort_values(
            ascending=False
        )
        .head(10)
        .reset_index()
    )

    top_movies = top_movies.merge(
        movies,
        on="movie_id"
    )

    st.dataframe(
        top_movies[
            [
                "title",
                "rating"
            ]
        ]
    )

# =====================================================
# ABOUT MODEL
# =====================================================

elif page == "About Model":

    st.title("🤖 Model Information")

    st.markdown("""
    ### Algorithm Used

    Singular Value Decomposition (SVD)

    ### Workflow

    User Ratings
        ↓

    User-Movie Matrix
        ↓

    Matrix Factorization
        ↓

    Latent Features
        ↓

    Predicted Ratings
        ↓

    Top-N Recommendations

    ### Performance

    RMSE ≈ 0.93

    MAE ≈ 0.73

    ### Dataset

    MovieLens 100K

    Users: 943

    Movies: 1682

    Ratings: 100,000
    """)