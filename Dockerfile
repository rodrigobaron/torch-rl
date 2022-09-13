FROM nvidia/cuda:11.4.2-runtime-ubuntu20.04

# install ubuntu dependencies
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get update && \
    apt-get -y install python3-pip \
    xvfb \
    ffmpeg \
    git \
    build-essential \
    cmake python-opengl \
    wget \
    unzip \
    software-properties-common \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev patchelf
RUN ln -s /usr/bin/python3 /usr/bin/python

# install python dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# install mujoco
RUN apt-get -y install 

COPY entrypoint.sh /usr/local/bin/
# RUN chmod 777 /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# copy local files
COPY . /app
WORKDIR /app