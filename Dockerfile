# Use a lightweight FastAPI-optimized image
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10-slim

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Set the working directory
WORKDIR /code

# Copy dependencies file and install required packages
COPY app/api_requirements.txt .
RUN pip install --no-cache-dir --upgrade -r api_requirements.txt

# Copy application and model files
COPY ./app /code/app
COPY ./app/weather_disease_model.pkl .
COPY ./app/minmax_scaler.pkl .
COPY ./app/label_encoder.pkl .

# Expose port
EXPOSE 9696

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9696"]