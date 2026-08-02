FROM python:3.11-slim

WORKDIR /app

# Dependencies first, so editing the code does not rebuild this layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY frontend/ frontend/
COPY .streamlit/ .streamlit/
COPY data/sample/ data/sample/

# Build the sample catalogue into the image, so `docker compose up` works from a
# fresh clone with nothing to download. Mount data/processed to use the full one.
RUN python -m app.build --sample

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000 8501

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
