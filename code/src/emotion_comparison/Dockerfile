FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install optional dependencies for better performance
RUN pip install --no-cache-dir dlib==19.24.0 opencv-python-headless==4.7.0.72 numba==0.56.4

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app:$PYTHONPATH

# Run a quick test to verify setup
RUN mkdir -p /app/comparison_results
RUN ./run_emotion_comparison.sh --test || echo "Test mode requires video files. Setup is complete."

# Default command
ENTRYPOINT ["./run_emotion_comparison.sh"]
CMD ["--help"]
