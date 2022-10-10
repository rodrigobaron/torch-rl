FROM rodrigobaron/torch-notebook:cuda-114

COPY . /app

RUN pip install -r requirements.txt