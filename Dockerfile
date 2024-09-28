FROM python:3.12

WORKDIR /app
ENV TZ=UTC

RUN apt-get update && apt-get install -y

COPY ./requirements.txt /app/
COPY ./src/data_refresh.py app/src/

RUN pip install --upgrade pip && pip install -r /app/requirements.txt
