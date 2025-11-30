# ----------------------------------------
# Base Image
# ----------------------------------------
FROM python:3.12-slim

# ----------------------------------------
# Install system dependencies for LightGBM & CatBoost
# ----------------------------------------
RUN apt-get update && apt-get install -y \
    libgomp1 \
    gcc \
    g++ \
    gfortran \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ----------------------------------------
# Set working directory
# ----------------------------------------
WORKDIR /app

# ----------------------------------------
# Copy requirements file
# ----------------------------------------
COPY requirements_docker.txt .

# ----------------------------------------
# Install dependencies
# ----------------------------------------
RUN pip install --no-cache-dir -r requirements_docker.txt

# ----------------------------------------
# Create folders (important for COPY)
# ----------------------------------------
RUN mkdir -p models \
    && mkdir -p scripts

# ----------------------------------------
# Copy application files
# ----------------------------------------
COPY app.py .

# ----------------------------------------
# Copy model and preprocessing utilities
# ----------------------------------------
COPY models/preprocessor.joblib models/preprocessor.joblib
COPY scripts/data_clean_utils.py scripts/data_clean_utils.py

# ----------------------------------------
# Expose FastAPI port
# ----------------------------------------
EXPOSE 8000

# ----------------------------------------
# Run the FastAPI app
# ----------------------------------------
CMD ["python", "app.py"]
