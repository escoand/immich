FROM python:3.11-slim

RUN pip install rknn-toolkit2
RUN apt-get update && \
    apt-get install -y libgl1 libglib2.0-0

COPY run.py /

ENTRYPOINT [ "python", "/run.py" ]
