FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend app
COPY app.py .

# Copy frontend build
COPY frontend-build ./frontend-build

# Expose port
EXPOSE 5000

# Run app
CMD ["python", "app.py"]