FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

WORKDIR /src

COPY requirements.txt .

RUN apt update && \
    apt install -y libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 nano wget && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

    RUN pip install -r requirements.txt --no-cache-dir
COPY . .

RUN mv ./configs/clearml.conf /root/clearml.conf
RUN clearml-init

CMD ["python", "main.py", "--config=./configs/config.yaml", "--output-dir=outputs", "--device=0"]
