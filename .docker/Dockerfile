# Use the Miniconda base image
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the required packages using conda and pip from requirements.txt
RUN conda install -c conda-forge --file requirements.txt

# Install test dependencies
RUN pip install pytest

# Run the tests when the container is started
CMD ["pytest", "tests/"]