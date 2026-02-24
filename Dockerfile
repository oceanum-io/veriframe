FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for cartopy and geopandas
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgeos-dev \
    libproj-dev \
    proj-data \
    proj-bin \
    libgdal-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip

# Copy the package source
COPY . .

# Install veriframe
RUN pip install --no-cache-dir .

# Default command
CMD ["python"]
