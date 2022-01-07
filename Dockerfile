FROM julianassmann/opencv-cuda:cuda-10.2-opencv-4.2


RUN apt update && \
    apt install --no-install-recommends -y build-essential software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install --no-install-recommends -y python3.6 python3-pip python3-setuptools python3-distutils && \
    apt clean && rm -rf /var/lib/apt/lists/* 



WORKDIR /app


RUN python3.6 -m pip install --upgrade pip && \
    python3.6 -m pip install numpy==1.19.5 && \
    python3.6 -m pip install Flask && \
    python3.6 -m pip install keras==2.6.0 && \
    python3.6 -m pip install matplotlib==3.3.4 && \
    python3.6 -m pip install opencv-python==4.5.5.62 && \
    python3.6 -m pip install Pillow==8.4.0 && \
    python3.6 -m pip install tensorflow-gpu==2.6.2 && \
    python3.6 -m pip install urllib3

COPY app/ .

ENV PYTHONUNBUFFERED=TRUE
# ENV SAGEMAKER_PROGRAM main.py
# ENTRYPOINT ["python3"]