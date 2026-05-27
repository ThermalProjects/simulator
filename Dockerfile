FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir websockets==12.0 numpy
COPY plant.py .
COPY ss_ws.py .
EXPOSE 5005
CMD ["python", "ss_ws.py"]
