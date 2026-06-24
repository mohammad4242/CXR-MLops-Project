# 1. Use a lightweight Python base image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install system dependencies if needed (often required by scikit-image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy the requirements file first (this helps Docker cache the installation)
COPY requirements.txt .

# 5. Install Python packages without saving cache to keep the image small
RUN pip install --no-cache-dir --timeout 120 --retries 5 -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 6. Copy your Python files (main.py, inference.py) into the container
COPY . .

# 7. Expose the port FastAPI uses
EXPOSE 8000

# 8. Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
