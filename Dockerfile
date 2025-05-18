FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime
WORKDIR /usr/src/app
COPY . .
# Dependencies
RUN apt update
RUN apt upgrade -y
RUN apt install -y libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 nano
RUN pip install -r requirements.txt --no-cache-dir
# Init ClearML
RUN mv ./configs/clearml.conf /root/clearml.conf
RUN clearml-init

CMD ["bash"]