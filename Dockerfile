# Fraud API image. Build:  docker build -t fraud-api .
# Run (secrets passed at RUN time, never baked in):
#   docker run --env-file .env -e HISTORY_BACKEND=neon -p 8000:8000 fraud-api

# 1. Start from a small official Python image (Debian "slim": no extras we don't need)
FROM python:3.13-slim

# 2. Python settings: no .pyc files, logs appear immediately, our package importable from src/.
#    The container has no Parquet files, so card history must come from Neon.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HISTORY_BACKEND=neon

WORKDIR /app

# 3. System library XGBoost needs (OpenMP), then clean the package lists to keep the layer small
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Dependencies FIRST: this slow layer is cached and reused until requirements-api.txt changes
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# 5. Then the things that change more often: settings, the model (baked in), the code
COPY params.yaml .
COPY models/model.json models/feature_spec.json models/guardrails.json models/
COPY src/ src/

# 6. Run as a normal user, not root: a break-in doesn't get admin rights inside the container
RUN useradd --create-home appuser && mkdir -p /app/logs && chown appuser /app/logs
USER appuser

# 7. Start the API on all network interfaces, on the port the platform gives us ($PORT on Render)
EXPOSE 8000
CMD ["sh", "-c", "uvicorn fraud.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
