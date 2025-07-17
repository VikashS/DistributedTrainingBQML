FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data_preparation.py .

ENTRYPOINT ["python", "etl.py"]