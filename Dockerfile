# Use Python 3.10 (TensorFlow 2.10 supports only 3.7–3.10)
FROM python:3.10-slim

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for librosa + scientific libs
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create working folder
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Streamlit default port
EXPOSE 8501

# Run streamlit app
CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0"]
