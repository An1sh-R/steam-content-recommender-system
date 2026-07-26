FROM python:3.11-slim

WORKDIR /app

# Dependencies first, so code edits do not invalidate the layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY recommender/ recommender/
COPY api/ api/
COPY app/ app/
COPY data/sample/ data/sample/

# Bake the sample catalogue in, so `docker compose up` works from a fresh
# clone with no build step. Mount data/processed to use the full dataset.
RUN python -m recommender.build --sample

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000 8501

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
