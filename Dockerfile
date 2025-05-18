FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime
WORKDIR /src
COPY . .
# Dependencies
RUN apt update && apt upgrade -y
RUN apt install -y libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 nano wget
RUN pip install -r requirements.txt --no-cache-dir
# Init ClearML
RUN mv ./configs/clearml.conf /root/clearml.conf
RUN clearml-init

CMD ["python", "train.py", "--config=./configs/train_test.yaml", "--output-dir=outputs", "--device=0"]