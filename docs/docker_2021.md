# Development for Data Scientist:
## Docker 

## Course
<iframe width="560" height="315" src="https://www.youtube.com/embed/loMf5bFyzY4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

*   [Slides](slides/Code_Development_Docker.pdf)
<!-- *   [Practical session](https://github.com/wikistat/AI-Frameworks/blob/master/CodeDevelopment/TP.pdf) -->

## Practical Session

In this practical session, you will now run your code through a Docker container.  
Using docker in data science projects has two advantages:
*   Improving the reproducibility of the results  
*   Facilitating the portability and deployment
In this session, we will try to package the code from the previous session, allowing us to train a neural network to colorize images into a Docker image and use this image to instantiate a container on a GCloud instance to run the code.

We will first create the Dockerfile corresponding to our environment.  

On your local machine, create a new file named `Dockerfile` containing the following code:
```python
# Base image from pytorch
FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
# Set up for your local zone an UTC information
ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
# Additional librairies
RUN pip install tqdm tensorboard
```

Take a moment to analyze this dockerfile.
As you can see, it is built upon an existing image from Pytorch.
Starting from existing images allows for fast prototyping. You may find existing images on [DockerHub](https://hub.docker.com/). The Pytorch image we will be using is available [here](https://hub.docker.com/r/pytorch/pytorch)


Fire up your GCloud instance and send your dockerfile using 
```console
gcloud compute scp [your_file_path] [your_instance_name]:Workspace/ --zone [your_instance_zone]
```

Connect to your instance:
```console
gcloud compute ssh --zone [your_instance_zone] [your_instance_name]
```

If docker is not already installed in your machine, follow [this guide](https://docs.docker.com/engine/install/) to install it.
You will also need the NVIDIA Container Toolkit to be installed to allow docker to communicate with the instance GPU.
If you created your GCloud instance following the previous session's instructions, it should be OK.
To verify that it is OK you may run the following command:

```console
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```
If the command output is in the form of :
```console
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.51.06    Driver Version: 450.51.06    CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            On   | 00000000:00:1E.0 Off |                    0 |
| N/A   34C    P8     9W /  70W |      0MiB / 15109MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```
then everything is OK. Otherwise, you may need to install the NVIDIA Container Toolkit manually, following [this guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

You may now build your first image using the following command:

```console
sudo docker build -t [your_image_name] -f [path_to_your_image]  [build_context_folder]
```

The image should take a few minutes to build.  
Once this is done, use the following command to list the available images on your GCloud instance:
```console
sudo docker image ls
```
How many images can you see? What do they refer to?  

Now that our images are built, we can now use them to instantiate containers.
Since a container is an instance of an image, we can instantiate several containers using a single image.

We will run our first container using the interactive mode.

Run the following command to run your fist container:
```console
docker run -it --name [your_container_name] [your_image_name]
```
You should now have access to an interactive terminal from your container.  
On this terminal, open a Python console and check that Pytorch is installed and has access to your instance GPU.
```python
import torch
print(torch.cuda.is_available())
```

Quit the Python console and quit your container using `ctrl+d`.  
You can list all your running containers using the following command:
```console
sudo docker container ls
```
Your container is closed and does not appear.
To list all the existing containers, add the ```-a``` to the previous command.
```console
sudo docker container ls -a
```

Start your containers using:
```console
sudo docker start [container_id_or_name]
```
Check that it is now listed as started.
You can have access to its interactive mode using the `attach` command:

```console
sudo docker attach [container_id_or_name]
```

You can delete a container using the `rm` command:
```console
sudo docker rm [container_id_or_name]
```

We will now see how to share data between the container and the machine it is running on.
First create a folder containing the files:
*   `download_landscapes.sh`
*   `unet.py`
*   `colorize.py`
*   `data_utils.py`

Create a new container using this time mounting a shared volume using the following command:
```console
docker run -it --name [container_name] -v ~/[your_folder_path]:/workspace/[folder_name] [image_name]
```
Go to the shared folder and run the `download_landscapes.sh` script.
Leave the container and look at your folder in the local. What can you see?

If you want to run your job using the interactive mode, you need to give access at your container to your host resources.

Start a new container using the following command to get access to the GPU and CPU resources and run the `colorize.py` script.
```console
docker run -it --gpus all --ipc=host --name [container_name] -v ~/[your_folder_path]:/workspace/[folder_name] [image_name]
```

Now try to send all the results and weights to your local machine and maybe to look at the tensorboard logs.
