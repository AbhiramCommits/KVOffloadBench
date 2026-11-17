# CUDA runtime image for the GPU host. The benchmark itself is Python.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    VLLM_USE_V1=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY kvoffloadbench ./kvoffloadbench
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
COPY Makefile ./
RUN python3 -m pip install --break-system-packages --no-cache-dir -e ".[gpu]"

COPY . .

ENTRYPOINT ["kvoffloadbench"]
CMD ["--help"]
