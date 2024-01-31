# Use an official lightweight Python image.
FROM python:3.9-slim

# Set the working directory in the container to /app.
WORKDIR /app

# Copy the contents of the repository into the container.
COPY . .

# Install the required packages using pip from the requirements.txt file.
# If requirements.txt does not include all the necessary packages, install them here.
RUN pip install --no-cache-dir scikit-learn==0.24.2 \
                                bokeh==2.3.3 \
                                numpy>=1.20.0 \
                                pandas>=1.2.0 \
                                scipy>=1.6.0 \
                                matplotlib>=3.3.0 \
                                seaborn>=0.11.0 \
                                pytest>=6.0.0

# For torch, you need to specify the correct version, here I'm assuming 1.8.0 which is compatible with CUDA 10.2.
# Adjust the version numbers as necessary.
RUN pip install --no-cache-dir torch==1.8.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Run the tests when the container launches.
CMD ["pytest", "tests/"]