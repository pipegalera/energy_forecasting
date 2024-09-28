FROM python:3.12-slim

WORKDIR /app
ENV TZ=UTC

RUN apt-get update && apt-get install -y

COPY .env ./
COPY requirements.txt ./
COPY src/data_refresh.py src/
COPY src/data_backfill.py src/

RUN pip install --upgrade pip && pip install -r /app/requirements.txt
CMD ["python", "src/data_refresh.py"]
