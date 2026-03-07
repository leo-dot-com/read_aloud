# ---- Builder stage (full Python image for compilation) ----
FROM python:3.11 AS builder

WORKDIR /app

# Install system dependencies required for building (ffmpeg not needed here)
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade pip and install a known‑good setuptools version
RUN pip install --no-cache-dir --upgrade pip setuptools==69.5.1 wheel

# Install torch and numpy first (pre‑built wheels)
RUN pip install --no-cache-dir torch==2.0.1 numpy==1.24.3

# Install openai-whisper directly from GitHub with the correct tag
# --no-build-isolation ensures it uses the venv's setuptools
RUN pip install --no-cache-dir --no-build-isolation git+https://github.com/openai/whisper.git@v20231117

# Copy requirements.txt and install remaining packages
COPY requirements.txt .
# Remove openai-whisper line (already installed) to avoid conflicts
RUN sed -i '/openai-whisper/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY whisper_api.py .

# ---- Final stage (slim image for runtime) ----
FROM python:3.11-slim

WORKDIR /app

# Install runtime system dependencies (ffmpeg for audio, curl for health checks)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire virtual environment from the builder
COPY --from=builder /opt/venv /opt/venv
# Copy the application code
COPY --from=builder /app/whisper_api.py .

# Set the PATH to use the venv's Python
ENV PATH="/opt/venv/bin:$PATH"

# Expose the port your app listens on
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application (as in the original Dockerfile)
CMD ["python", "whisper_api.py"]
