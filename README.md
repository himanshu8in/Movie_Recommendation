# 🎬 MovieLens Hybrid Recommendation System

A machine learning-powered movie recommendation system built using the MovieLens 100K dataset. This project combines **Collaborative Filtering (SVD)** and **Content-Based Filtering** to generate personalized movie recommendations for users.

---

## 📌 Project Overview

Recommendation systems are widely used by platforms such as Netflix, Amazon, YouTube, and Spotify to improve user engagement through personalized suggestions.

This project implements:

- Collaborative Filtering using Singular Value Decomposition (SVD)
- Content-Based Filtering using movie metadata
- Hybrid Recommendation System
- Model Evaluation using RMSE and MAE
- Interactive Streamlit Dashboard

---

## 🚀 Features

✅ Personalized movie recommendations

✅ SVD-based Collaborative Filtering

✅ Content-Based Movie Similarity

✅ Hybrid Recommendation Engine

✅ Top-N Recommendations

✅ Model Evaluation Metrics

✅ Interactive Streamlit Web Application

✅ Modular Project Structure

---

## 📂 Project Structure

```text
MovieLens-Recommender/
│
├── data/
│   ├── u.data
│   ├── u.item
│   └── u.user
│
├── models/
│   └── svd_model.pkl
│
├── notebooks/
│   ├── EDA.ipynb
│   └── Model_Training.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── collaborative.py
│   ├── content_based.py
│   ├── hybrid.py
│   └── evaluation.py
│
├── train.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Dataset

This project uses the MovieLens 100K dataset.

### Dataset Statistics

| Metric | Value |
|----------|----------|
| Users | 943 |
| Movies | 1,682 |
| Ratings | 100,000 |
| Rating Scale | 1 - 5 |
| Sparsity | ~93.7% |

Dataset Source:

:contentReference[oaicite:0]{index=0}

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|----------|
| Python | Core Development |
| Pandas | Data Manipulation |
| NumPy | Numerical Computation |
| Scikit-Learn | Similarity Computation |
| Surprise | Collaborative Filtering |
| Streamlit | Web Application |
| Matplotlib | Visualization |
| Seaborn | Exploratory Data Analysis |
| Pickle | Model Persistence |

---

## 🔍 Exploratory Data Analysis

The following analyses were performed:

- Rating Distribution
- User Activity Analysis
- Movie Popularity Analysis
- Sparsity Analysis
- User-Movie Matrix Generation
- Top Rated Movies
- Most Rated Movies
- Demographic Analysis

### Key Findings

- Dataset contains 100,000 ratings.
- User-Movie matrix is approximately 93.7% sparse.
- Ratings are skewed towards positive feedback.
- A small subset of movies receives the majority of ratings.
- High sparsity motivates the use of matrix factorization techniques.

---

## 🤖 Recommendation Models

### 1. Collaborative Filtering

Implemented using Singular Value Decomposition (SVD).

SVD decomposes the user-item interaction matrix into latent factors that capture hidden relationships between users and movies.

Advantages:

- Handles sparse datasets effectively
- Learns latent user preferences
- Provides highly personalized recommendations

---

### 2. Content-Based Filtering

Implemented using:

- TF-IDF Vectorization
- Cosine Similarity

Movies are recommended based on similarity between movie features.

Advantages:

- Works for users with limited interaction history
- Explainable recommendations

---

### 3. Hybrid Recommendation System

Final recommendation score:

```text
Hybrid Score =
0.7 × Collaborative Score
+
0.3 × Content Similarity Score
```

This combines the strengths of both approaches while reducing their individual limitations.

---

## 📈 Model Evaluation

The recommendation model was evaluated using:

### Root Mean Squared Error (RMSE)

Measures prediction error between actual and predicted ratings.

```text
RMSE ≈ 0.93
```

### Mean Absolute Error (MAE)

Measures average prediction deviation.

```text
MAE ≈ 0.73
```

### Cross Validation

5-Fold Cross Validation was performed to ensure model robustness and generalization.

---

## 🎯 Sample Recommendations

```text
User ID: 10

1. Schindler's List
2. Casablanca
3. Close Shave, A
4. Shawshank Redemption
5. Star Wars
6. Godfather
7. Rear Window
8. Silence of the Lambs
9. Good Will Hunting
10. Usual Suspects
```

---

## ▶️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/MovieLens-Recommender.git

cd MovieLens-Recommender
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows

```bash
venv\Scripts\activate
```

Linux/Mac

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ Train Model

Run:

```bash
python train.py
```

This will:

- Load dataset
- Train SVD model
- Evaluate performance
- Save model as:

```text
models/svd_model.pkl
```

---

## 🌐 Launch Streamlit Application

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

## 📷 Application Screenshots

### Home Page

```text
Add Screenshot Here
```

### Recommendations Page

```text
Add Screenshot Here
```

### Analytics Dashboard

```text
Add Screenshot Here
```

---

## 🔮 Future Improvements

- Deep Learning Recommendation Models
- Neural Collaborative Filtering (NCF)
- Implicit Feedback Integration
- Real-Time Recommendation API
- Docker Deployment
- Cloud Deployment
- Recommendation Explainability
- User Authentication System

---

## 📚 Learning Outcomes

Through this project, I gained practical experience in:

- Recommendation Systems
- Matrix Factorization
- Collaborative Filtering
- Content-Based Filtering
- Machine Learning Evaluation Metrics
- Model Deployment
- Streamlit Application Development

---

## 👨‍💻 Author

**Himanshu Singh**

Artificial Intelligence & Machine Learning Undergraduate

GitHub: https://github.com/himanshu8in

LinkedIn: https://www.linkedin.com/in/himanshu8in/

---

## ⭐ If you found this project useful, consider giving it a star.