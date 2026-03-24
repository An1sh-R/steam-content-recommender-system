FROM python:3.11-slim

WORKDIR /app

# copy requirements first (for caching)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# copy project
COPY . .

# expose FastAPI port
EXPOSE 8000

# run server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "10000"]