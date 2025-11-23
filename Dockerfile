FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
FROM python:3.9-slim-buster

WORKDIR /test_env
COPY /src* /test_env/src

RUN pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118


CMD ["python", "src/test_pop.py"]



