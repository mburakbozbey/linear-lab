# Use an official lightweight Python image.
FROM python:3.8.18

# Set environment variables to optimize Python execution.
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container to /app.
WORKDIR /app

# Copy the local directory contents into the container's /app directory.
COPY . /app

# Install the required packages using pip from the requirements.txt file.
RUN pip install --no-cache-dir -r requirements.txt

# Set the PYTHONPATH environment variable to /app.
ENV PYTHONPATH /app

# Run the tests as part of the Docker build.
RUN pytest tests/