# Dockerfile.etl
# Use a slim Python 3.9 image as a base
FROM python:3.9-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the Python dependencies
# --no-cache-dir: Prevents pip from storing packages in cache, reducing image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy your entire 'fraud' Python package into the container
# This makes your fraud/etl/data_preparation.py accessible
COPY fraud ./fraud

# Define the default command to run when the container starts.
# It executes your data_preparation.py script.
ENTRYPOINT ["python", "fraud/etl/data_preparation.py"]