FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip3 install --no-cache-dir -r requirements.txt

# Download the model
RUN mkdir -p /root/.cache/huggingface/hub/models--TheBloke--zephyr-7B-beta-GGUF/snapshots/174361791a82333121d28aa293297a7d532af18d/
RUN wget https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q5_K_M.gguf -O /root/.cache/huggingface/hub/models--TheBloke--zephyr-7B-beta-GGUF/snapshots/174361791a82333121d28aa293297a7d532af18d/zephyr-7b-beta.Q5_K_M.gguf

# Update creation_params.json to point to the downloaded model
RUN sed -i 's|\"model_path\": \".*\"|\"model_path\": \"/root/.cache/huggingface/hub/models--TheBloke--zephyr-7B-beta-GGUF/snapshots/174361791a82333121d28aa293297a7d532af18d/zephyr-7b-beta.Q5_K_M.gguf\"|' creation_params.json

CMD ["python3", "ai_voicetalk_local.py"]
