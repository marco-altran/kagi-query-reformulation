FROM ubuntu:24.04 AS builder
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential cmake git python3 python3-pip pkg-config libopenblas-dev libcurl4-openssl-dev \
 && rm -rf /var/lib/apt/lists/*

FROM ubuntu:24.04
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Python + venv
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
     python3 \
     python3-venv \
     python3-dev \
     build-essential \
     libstdc++6 \
  && rm -rf /var/lib/apt/lists/

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    flask \
    pydantic \
    openai

COPY .env .
COPY api_cloud.py .

EXPOSE 8080

CMD ["uvicorn", "api_cloud:app", "--host", "0.0.0.0", "--port", "8080"]
