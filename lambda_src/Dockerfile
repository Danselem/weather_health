FROM public.ecr.aws/lambda/python:3.10

# Copy requirements and install
COPY requirements.txt .
RUN pip install -r requirements.txt --target /opt/python

# Copy model & code
COPY weather_disease_model.pkl label_encoder.pkl feature_order.json /opt/model/
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

ENV MODEL_DIR=/opt/model

# Set the handler
CMD ["lambda_function.lambda_handler"]
