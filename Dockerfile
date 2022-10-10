#FROM nvidia/cuda:11.7.1-base-ubuntu20.04
FROM nvidia/cuda:11.4.2-runtime-ubuntu20.04

LABEL maintainer="Rodrigo Baron <baron.rodrigo0@gmail.com>"
LABEL repository="torch-jupyter"

# Set bash as default shell
ENV SHELL=/bin/bash
ENV SYS_MEM_LIMIT=16
ENV SYS_CPU_LIMIT=8

# Create a working directory
WORKDIR /app/

# Set the locale
RUN apt-get update && apt-get -y install curl locales
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8

# Build with some basic utilities
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get update && apt-get -y install \
    python3-pip \
    xvfb \
    ffmpeg \
    apt-utils \
    vim \
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

# alias python='python3'
RUN ln -s /usr/bin/python3 /usr/bin/python

# setup base packages
RUN pip install \
    numpy==1.23.1 \
    torch==1.12.0 \
    jupyterlab==3.4.7 \
    pandas \
    ipywidgets

# install  nodejs 18.X for extensions install
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
RUN apt-get install -y nodejs

# install and setup theme extensions
RUN jupyter labextension install @telamonian/theme-darcula
COPY settings/themes.jupyterlab-settings /root/.jupyter/lab/user-settings/\@jupyterlab/apputils-extension/themes.jupyterlab-settings

# install and setup lsp server and extension. It provide code completion
# and documentation inspect along with flake8 hints
RUN pip install jupyterlab-lsp
RUN pip install 'python-lsp-server[all]'
COPY settings/completion.jupyterlab-settings /root/.jupyter/lab/user-settings/\@krassowski/jupyterlab-lsp/completion.jupyterlab-settings 
RUN jupyter serverextension enable --py jupyter_lsp

# setup system monitor
RUN pip install jupyterlab-system-monitor
COPY settings/jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py

# setup execution time cells and line numbers
RUN pip install jupyterlab_execute_time
COPY settings/tracker.jupyterlab-settings /root/.jupyter/lab/user-settings/\@jupyterlab/notebook-extension/tracker.jupyterlab-settings

# additional extensions
RUN pip install jupyterlab-git
RUN jupyter labextension install @jupyterlab/latex
RUN jupyter labextension install @jupyterlab/debugger
RUN pip install aquirdturtle_collapsible_headings

COPY entrypoint.sh /usr/local/bin/

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
EXPOSE 8888
