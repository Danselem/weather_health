# Use a lightweight FastAPI-optimized image
FROM python:3.10-slim

# Set the working directory
WORKDIR /code

# Copy requirements and install
COPY service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service files
COPY service/app.py .
COPY models/weather_disease_model.pkl .
COPY data/processed/minmax_scaler.pkl .
COPY data/processed/label_encoder.pkl .

# Expose port
EXPOSE 8080

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]