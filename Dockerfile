# Reproducible research environment — one command reruns every result.
#   docker build -t nwm .
#   docker run --rm -v $(pwd)/out:/app/assets nwm            # full pipeline
#   docker run --rm nwm python tests/smoke_test.py           # fast check
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY scripts/ scripts/
COPY tests/ tests/
COPY models_pytorch/ models_pytorch/
COPY ros2_ws/ ros2_ws/

ENV PYTHONPATH=/app/src
CMD ["python", "scripts/run_pipeline.py"]
