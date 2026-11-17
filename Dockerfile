# CUDA runtime image for the GPU host. The benchmark itself is Python.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    VLLM_USE_V1=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common git ca-certificates \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
    && python3.11 -m venv /opt/venv \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /app
COPY pyproject.toml README.md ./
COPY kvoffloadbench ./kvoffloadbench
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
COPY Makefile ./
RUN pip install --no-cache-dir -e ".[gpu]"

COPY . .

ENTRYPOINT ["kvoffloadbench"]
CMD ["--help"]
