FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app MPLBACKEND=Agg
WORKDIR /app
COPY wendy/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY . .
RUN python -m utils.weak_sources wendy
CMD ["python", "clean_run_metrics.py", "wendy", "--protocol", "native"]
