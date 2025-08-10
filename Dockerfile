FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Clone Sonic repository
RUN git clone https://github.com/jixiaozhong/Sonic.git

# Install Python dependencies
WORKDIR /workspace/Sonic
COPY requirements_custom.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements_custom.txt && \
    pip install runpod && \
    rm -rf /root/.cache/pip

# Download model weights using huggingface-cli
RUN pip install huggingface_hub[cli] && \
    huggingface-cli download LeonJoe13/Sonic --local-dir checkpoints && \
    huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt --local-dir checkpoints/stable-video-diffusion-img2vid-xt && \
    huggingface-cli download openai/whisper-tiny --local-dir checkpoints/whisper-tiny

# Copy the RunPod handler
COPY handler.py /workspace/handler.py

# Set environment variables
ENV PYTHONPATH="/workspace/Sonic:$PYTHONPATH"
ENV CUDA_VISIBLE_DEVICES="0"

# RunPod expects the handler at /workspace/handler.py
WORKDIR /workspace

# Start the serverless handler
CMD ["python", "-u", "handler.py"]
