# Use an official Python runtime as a parent image
FROM python:3.9.5

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install additional Python packages
RUN pip install -U sentence-transformers langchain-community langchain huggingface-hub transformers streamlit

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variable
ENV NAME World

# Install curl for connectivity tests
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Check internet connectivity
RUN curl -s https://www.google.com > /dev/null || (echo "Internet connectivity issue detected" && exit 1)

# Run streamlit when the container launches
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
