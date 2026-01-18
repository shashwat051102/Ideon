FROM python:3.11-slim

# Ensure logs are unbuffered (show up instantly in EB logs)
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose HTTP port used by the app (matched by Nginx upstream)
EXPOSE 5000

# Production server: single worker to avoid SQLite/Chroma locking issues
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "run:app"]
