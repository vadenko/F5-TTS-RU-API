#!/bin/bash
set -e

if [ -n "${HUGGINGFACE_HUB_TOKEN}" ]; then
  mkdir -p /root/.huggingface
  echo -n "${HUGGINGFACE_HUB_TOKEN}" > /root/.huggingface/token
  export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN}"
fi

mkdir -p /data/input /data/output /data/books /data/temp

MODEL_REPO="Misha24-10/F5-TTS_RUSSIAN"
MODEL_CACHE="/root/.cache/huggingface/hub"

python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${MODEL_REPO}', cache_dir='${MODEL_CACHE}')"

export PYTHONPATH=/app:$PYTHONPATH
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-4123} --workers 1
