FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN useradd -m -u 10001 app
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY --chown=app:app . .
# corpus/ and emb_cache/ are mounted or baked into a PRIVATE image; secrets come from env vars, never the image
RUN mkdir -p /app/.data && chown app:app /app/.data
USER app
EXPOSE 8080
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
