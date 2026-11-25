FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app MPLBACKEND=Agg
WORKDIR /app
COPY wsindy/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /tmp/requirements.txt
COPY . .
RUN python -m utils.weak_sources wsindy
CMD ["python", "clean_run_metrics.py", "wsindy"]
