# Use an official lightweight Python image.
FROM python:3.9-slim

# Set the working directory in the container to /app.
WORKDIR /app

# Copy the contents of the repository into the container.
COPY . .

# Install the required packages using pip from the requirements.txt file.
RUN pip install --no-cache-dir -r requirements.txt

# For torch, you need to specify the correct version, here I'm assuming 1.8.0 which is compatible with CUDA 10.2.
# Adjust the version numbers as necessary.
RUN pip install --no-cache-dir torch==1.8.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Run the tests when the container launches.
CMD ["pytest", "tests/"]