# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    python3-opencv \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements.txt first to leverage Docker cache
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of the application code
COPY . /app

# Expose port 8080 for Flask
EXPOSE 8080

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "app:app"]
