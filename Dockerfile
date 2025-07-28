FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements_dashboard.txt .
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_dashboard.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data results logs web_dashboard/assets

# Set environment variables
ENV PYTHONPATH=/app
ENV DASH_DEBUG=False
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8050

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8050 || exit 1

# Run the dashboard
CMD ["python", "web_dashboard/app.py"]