# Use official Python 3.11 slim image
FROM python:3.11-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8501

# Set work directory
WORKDIR /app

# Install system dependencies (build-essential for compiling if needed, libgomp1 for XGBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition and install packages
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY . /app/

# Expose Streamlit port
EXPOSE 8501

# Healthcheck to verify the web app is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the app
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
