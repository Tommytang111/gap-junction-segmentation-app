FROM python:3.10-slim

#Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

#Copy source code
COPY . .

#Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

#Create directories for models and uploads
RUN mkdir -p /app/uploads /app/outputs

#Expose port
EXPOSE 8000

#Run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
