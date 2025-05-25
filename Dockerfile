# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements
RUN pip install --no-cache-dir numpy pandas matplotlib seaborn scikit-learn shap

# Set the default command to run the script
CMD ["python", "one_neuron_net.py"] 
