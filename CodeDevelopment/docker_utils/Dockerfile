# Base image from tensorflow
FROM tensorflow/tensorflow:2.1.0-gpu-py3
# Set up for your local zone an UTC information
ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
#
RUN apt-get update && apt-get install -y python-opencv python-tk unzip pv
#
RUN pip install h5py scikit-image tqdm