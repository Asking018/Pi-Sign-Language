FROM python:3.9

# System dependencies for OpenCV / MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . /code

# App lives inside this folder in your repo
WORKDIR /code/Pi-Sign-Language-main

# Hugging Face Spaces runs on port 7860
ENV PORT=7860

CMD ["python", "app.py"]
