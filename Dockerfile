FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY rag_eval ./rag_eval
COPY data ./data
RUN pip install --no-cache-dir -e . && useradd -m app
USER app
ENV PORT=8000 PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/healthz')"
CMD ["rag-eval", "serve"]
