# Use Python base image
FROM python:3.12

# Install system dependencies required by OpenCV (libGL and GLib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt /app/
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

# Expose the Flask default port
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]