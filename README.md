# 🎮 Game Recommender System

A full-stack **hybrid video game recommender system** that combines content-based filtering with player profiling to deliver personalized game suggestions.

Deployed as a production-ready web application using modern ML + backend + cloud infrastructure.

---

## 🚀 Live Demo

* 🌐 Frontend: https://game-recommender-system.vercel.app
* ⚙️ Backend API: https://game-recommender-system-k9qi.onrender.com/docs

---

## 🧠 Features

* 🎯 **Content-Based Recommendation**

  * Uses TF-IDF vectorization on game metadata (genres, tags, descriptions)
  * Computes similarity via cosine similarity

* 🧩 **Player Profiling (Cold Start Solution)**

  * 10-question quiz maps user preferences to gameplay traits
  * Traits include: exploration, story, challenge, strategy, social, relaxation

* 🔀 **Hybrid Recommendation System**

  * Combines:

    * Content similarity
    * Game popularity
    * Player preference match
  * Produces a final hybrid ranking score

* ⚡ **Redis Caching**

  * Caches recommendations to reduce latency
  * Improves performance for repeated queries

* 👤 **User Authentication**

  * Secure login/register system
  * Passwords hashed using **bcrypt**
  * Persistent user profiles stored in PostgreSQL

* 🖼️ **Game UI Enhancements**

  * Dynamic Steam images using AppID
  * Fallback images for missing assets

---

## 🏗️ Tech Stack

### Backend

* FastAPI
* PostgreSQL (Supabase)
* Redis (Upstash)
* Docker

### Machine Learning

* TF-IDF (Scikit-learn)
* Cosine Similarity
* Custom hybrid scoring algorithm

### Frontend

* HTML, CSS, JavaScript
* Deployed on Vercel

---

## ⚙️ System Architecture

```text
Frontend (Vercel)
        ↓
FastAPI Backend (Render)
        ↓
PostgreSQL (Supabase) + Redis (Upstash)
        ↓
ML Recommendation Engine
```

---

## 📊 Recommendation Logic

### 1. Content-Based Filtering

* TF-IDF vectorization of:

  * Genres
  * Tags
  * Game descriptions
* Cosine similarity used to find similar games

### 2. Player Profile Matching

* Quiz answers → trait vector
* Compared with game trait vectors using cosine similarity

### 3. Hybrid Scoring

```text
Hybrid Score =
  0.6 × Content Similarity +
  0.2 × Popularity +
  0.2 × Player Match
```

---

## 📦 API Endpoints

### 🔐 Authentication

* `POST /auth/register`
* `POST /auth/login`

### 🎮 Recommendations

* `POST /recommend/game`
* `POST /recommend/quiz`

Interactive docs available at `/docs`

---

## 🧪 Example Request

```json
{
  "game": "witcher 3",
  "user_id": 1,
  "quiz_answers": [5,4,3,5,2,1,4,5,3,4],
  "top_n": 5
}
```

---

## 🛠️ Local Setup

### 1. Clone repo

```
git clone https://github.com/your-username/game-recommender-system.git
cd game-recommender-system
```

### 2. Create `.env`

```
DATABASE_URL=postgresql://user:pass@localhost:5432/db
REDIS_URL=redis://localhost:6379/0
```

### 3. Run with Docker

```
docker-compose up --build
```

### 4. Access

```
http://localhost:8000/docs
```

---

## ⚠️ Notes

* Uses environment variables for production compatibility
* Automatically switches between local and deployed environments
* Designed to handle cold-start users effectively

---

## 🎯 Future Improvements

* Collaborative filtering (user-user / item-item)
* Embedding-based recommendations (BERT / sentence transformers)
* Search autocomplete
* Better ranking optimization

---

## 👨‍💻 Author

Anish Ray

---

## ⭐ Acknowledgements

* Steam dataset (Kaggle)
* Inspired by Quantic Foundry player modeling approach
