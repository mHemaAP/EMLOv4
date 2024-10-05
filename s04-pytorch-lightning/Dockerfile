# Use the official Python image as a base
FROM python:3.9-slim

# Set the working directory
WORKDIR /

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# --no-cache-dir
# Copy the source code into the container
COPY . .
