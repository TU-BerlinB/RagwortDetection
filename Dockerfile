FROM python:3.12-slim

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/ /requirements/

ARG REQUIREMENTS_FILE

RUN pip install --no-cache-dir -r /requirements/${REQUIREMENTS_FILE}


CMD ["bash"]