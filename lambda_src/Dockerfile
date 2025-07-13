FROM public.ecr.aws/lambda/python:3.10

# Copy requirements and install
COPY requirements.txt .
RUN pip install -r requirements.txt --target /opt/python

# Copy model & code
COPY weather_disease_model.pkl minmax_scaler.pkl label_encoder.pkl /opt/model/
COPY predictor.py ${LAMBDA_TASK_ROOT}

# Set the handler
CMD ["predictor.handler"]
