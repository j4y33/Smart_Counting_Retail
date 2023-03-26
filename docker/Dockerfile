FROM nvcr.io/nvidia/tensorflow:19.10-py3

USER root


RUN apt-get update && apt-get -y --no-install-recommends install \
    sudo \
    vim \
    wget \
    build-essential \
    pkg-config \
    python3.6 \
    python3-pip \
    python3.6-dev \
    python3.6-venv \
    python-dev \
    python3-dev \
    ffmpeg


RUN apt-get -y --no-install-recommends install \
    git \
    cmake \
    autoconf \
    automake \
    libtool \
    gstreamer-1.0 \
    gstreamer1.0-dev \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-doc \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-alsa \
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 \
    gstreamer1.0-pulseaudio \
    python-gst-1.0 \
    libgirepository1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libcairo2-dev \
    gir1.2-gstreamer-1.0 \
    python3-gi \
    python-gi-dev

RUN mkdir -p /workspace
RUN mkdir -p /storage
RUN mkdir -p /dropbox

RUN apt install python3-pip
COPY requirements.txt /requirements.txt
RUN /usr/bin/python3 -m pip install -r /requirements.txt
RUN /usr/bin/python3 -m pip install --ignore-installed "dask[bag]" --upgrade
RUN /usr/bin/python3 -m pip install --ignore-installed "dask[distributed]" --upgrade
RUN /usr/bin/python3 -m pip install install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html
RUN cd / && git clone https://${username}:${password}@github.com/KaiyangZhou/deep-person-reid.git
RUN /usr/bin/python3 -m pip install -r /deep-person-reid/requirements.txt
RUN cd /deep-person-reid && /usr/bin/python3 setup.py develop

# ENV year 2021
# ENV month 1
# ENV day 1
# ENV camera entrance
# ENV video_folder /workspace/video-record/dvr/minio/minisio/

#RUN /workspace/camai/smart-counter-solid/smart-counter-reid/run.sh ${year} ${month} ${day} ${camera} ${video_folder}
