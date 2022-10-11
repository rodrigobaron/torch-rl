FROM rodrigobaron/torch-notebook:cuda-114

COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

ADD . /app
RUN pip install -e .