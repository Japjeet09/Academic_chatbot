FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --fix-missing build-essential default-mysql-client portaudio19-dev libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code and login image
COPY main_final.py .
COPY login.jpg .

# Expose Streamlit default port
EXPOSE 8501

CMD ["streamlit", "run", "main_final.py", "--server.port=8501"]
