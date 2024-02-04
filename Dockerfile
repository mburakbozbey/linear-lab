# Use an official lightweight Python image.
FROM python:3.8.18

# Set the working directory in the container to /app.
WORKDIR /app

# Set the PYTHONPATH environment variable
ENV PYTHONPATH=/app

# Copy the requirements.txt file into the container.
COPY requirements.txt ./

# Install the required packages using pip from the requirements.txt file.
RUN pip install --no-cache-dir -r requirements.txt

# For torch, you need to specify the correct version, here I'm assuming 1.8.0 which is compatible with CUDA 10.2.
# Adjust the version numbers as necessary.
RUN pip install --no-cache-dir torch==1.8.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Copy the rest of the application into the container.
COPY . .

# Run the tests as part of the Docker build.
RUN pytest tests/