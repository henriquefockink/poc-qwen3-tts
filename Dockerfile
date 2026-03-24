ARG VLLM_OMNI_VERSION=v0.16.0
FROM vllm/vllm-omni:${VLLM_OMNI_VERSION}

# Flash attention com SM configurável (para quem quiser rebuild)
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# Instalar qwen-tts (utilitários extras)
RUN pip install -U qwen-tts

# Copiar configs e scripts
COPY stage_configs/ /app/stage_configs/
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

WORKDIR /app
ENTRYPOINT ["/app/entrypoint.sh"]
