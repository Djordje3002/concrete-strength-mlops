FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --no-cache-dir . && useradd --uid 10001 --create-home appuser
COPY --chown=10001:10001 artifacts/model.joblib ./artifacts/model.joblib
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready')"
CMD ["uvicorn", "concrete_ml.api:app", "--host", "0.0.0.0", "--port", "8000"]
