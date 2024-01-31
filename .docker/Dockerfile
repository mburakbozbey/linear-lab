# Use the Miniconda base image
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Create a conda environment with the required packages
RUN conda create -n myenv python=3.9 \
    scikit-learn=0.24.2 \
    bokeh=2.3.3 \
    numpy>=1.20.0 \
    pandas>=1.2.0 \
    scipy>=1.6.0 \
    matplotlib>=3.3.0 \
    seaborn>=0.11.0 \
    pytest>=6.0.0 \
    pytorch>=1.8.0 -c pytorch

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Ensure the environment is activated:
RUN echo "Make sure flask is installed:"
RUN python -c "import flask"

# The code to run when container is started:
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "myenv", "python", "your_script.py"]
