# ==========================================
# STAGE 1: Builder
# ==========================================
FROM python:3.10-slim as builder

# Set working directory
WORKDIR /app

# (Optional but recommended) Install system build tools if any Python packages need compiling
RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential && rm -rf /var/lib/apt/lists/*

# Create a Python virtual environment
# We install dependencies here so we can easily copy them to the next stage
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ==========================================
# STAGE 2: Runner (The Final Image)
# ==========================================
FROM python:3.10-slim

# 1. Create a non-root user and group named 'appuser'
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# 2. Copy ONLY the installed dependencies from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Make sure the container uses the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# 3. Copy your application code and give ownership to 'appuser'
COPY --chown=appuser:appuser main.py .

# (Note: If you have your feature_repo or models folder, copy them like this too:)
# COPY --chown=appuser:appuser feature_repo/ ./feature_repo/

# 4. Switch from the default 'root' user to our restricted 'appuser'
USER appuser

# Tell Docker which port the container should listen on
EXPOSE 8000

# Start the Uvicorn server
ENTRYPOINT ["uvicorn"] 
CMD ["main:app", "--host", "0.0.0.0", "--port", "8000"]