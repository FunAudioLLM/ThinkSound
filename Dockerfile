FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    git-lfs \
    ffmpeg \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone the ThinkSound repository
RUN git clone https://github.com/liuhuadai/ThinkSound.git .

RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
RUN pip install -r requirements.txt

# Create a non-root user
RUN useradd -m -u 1000 thinksound && \
    chown -R thinksound:thinksound /app
USER thinksound

RUN chmod +x scripts/demo.sh

# Expose port for Gradio web interface
EXPOSE 7860

# Set default command to launch the web interface
CMD ["python", "app.py"]

# Alternative commands (uncomment as needed):
# For interactive bash session:
# CMD ["/bin/bash"]

# For running demo script (requires arguments):
# ENTRYPOINT ["./scripts/demo.sh"]
