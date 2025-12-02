FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "src.app.app:app", "--host", "0.0.0.0", "--port", "8000"]
