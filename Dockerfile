FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NYAI_ENV=production

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN adduser --disabled-password --gecos "" nyai
USER nyai

# Run the application
CMD ["gunicorn", "run:app", "--bind", "0.0.0.0:8080", "--workers", "4", "--threads", "2", "--timeout", "60"]

# Expose port
EXPOSE 8080 