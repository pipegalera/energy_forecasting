FROM python:3.12-slim

WORKDIR /app
ENV TZ=UTC

RUN apt-get update && apt-get install -y

COPY requirements.txt ./
COPY src ./src
COPY models ./models
COPY run_scripts.sh ./


RUN pip install --upgrade pip && pip install -r /app/requirements.txt
RUN chmod +x ./run_scripts.sh
