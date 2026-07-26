"""FastAPI application."""

from fastapi import FastAPI

from api.routes import router

app = FastAPI(
    title="Game Recommender API",
    description="Content-based Steam game recommendations.",
    version="2.0.0",
)
app.include_router(router)
