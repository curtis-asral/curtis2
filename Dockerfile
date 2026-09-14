# Use a lightweight Python base image
FROM python:3.12-slim

# Set a working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt /app/
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

# Expose the Flask default port
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
