# Use the official Python image as a base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code into the container
COPY . .

# Expose the port that your application will run on
EXPOSE 8181

# Set environment variables (optional)
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "telegram.py"]