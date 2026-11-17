FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY tokenwise ./tokenwise

ENV TOKENWISE_HOST=0.0.0.0 \
    TOKENWISE_PORT=8787 \
    TOKENWISE_DB=/data/tokenwise.db
VOLUME /data
EXPOSE 8787

CMD ["python", "-m", "tokenwise.main"]
