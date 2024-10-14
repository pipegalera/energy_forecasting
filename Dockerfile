FROM python:3.12-slim

WORKDIR /app
ENV TZ=UTC

RUN apt-get update && apt-get install -y

COPY requirements.txt ./
COPY src ./src
COPY run_scripts.sh ./


RUN pip install --upgrade pip && pip install -r /app/requirements.txt

CMD ["./run_scripts.sh"]
