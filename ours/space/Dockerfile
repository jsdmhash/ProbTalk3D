FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH HOST=0.0.0.0 PORT=7860 \
    OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
COPY --chown=user app.py .
EXPOSE 7860
CMD ["python", "app.py"]
