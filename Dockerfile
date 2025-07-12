# Use official Python image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements if available
COPY requirements.txt .

# Install dependencies if requirements.txt exists
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/

# Set default command
CMD ["python", "src/train.py"]