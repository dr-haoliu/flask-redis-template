FROM python:3.9.6-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy requirements from local to docker image
COPY requirements.txt /app

# Install the dependencies in the docker image
RUN pip3 install -r requirements.txt --no-cache-dir
RUN pip3 install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_lg-0.5.0.tar.gz

# Copy everything from the current dir to the image
COPY ./src .
