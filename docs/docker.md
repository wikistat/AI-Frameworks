# Development for Data Scientist:
## Docker 

## Course
<iframe width="560" height="315" src="https://www.youtube.com/embed/loMf5bFyzY4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

*   [Slides](https://github.com/wikistat/AI-Frameworks/tree/master/slides/Code_Development_Docker.pdf)


## Practical Session

In this practical session, you will now run your code through a Docker container.  
Using docker in data science projects has two advantages:  

*   Improving the reproducibility of the results  
*   Facilitating the portability and deployment  

In this session, we will try to package the code from our Gradio applications, allowing us to predict digits labels and to colorize images into a Docker image.  
We will then use this image to instantiate a container that could be hosted on any physical device to run the app.

We will first create the Dockerfile corresponding to our environment.  

On your local machine, create a new file named `Dockerfile` containing the following code:
```python
# Base image from pytorch
FROM pytorch/pytorch
# Set up for your local zone an UTC information
ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
# Additional librairies
RUN pip install gradio tensorboard
```

Take a moment to analyze this dockerfile.  
As you can see, it is built upon an existing image from Pytorch.  
Starting from existing images allows for fast prototyping. You may find existing images on [DockerHub](https://hub.docker.com/).  
The Pytorch image we will be using is available [here](https://hub.docker.com/r/pytorch/pytorch).


If docker is not already installed in your machine, follow [this guide](https://docs.docker.com/engine/install/) to install it.

You may now build your first image using the following command:

```console
sudo docker build -t [your_image_name] -f [path_to_your_image]  [build_context_folder]
```

The image should take a few minutes to build.  
Once it is done, use the following command to list the available images on your device:
```console
sudo docker image ls
```
How many images can you see? What do they refer to?  

Now that our images are built, we can use them to instantiate containers.
Since a container is an instance of an image, we can instantiate several containers using a single image.

We will run our first container using the interactive mode.

Run the following command to run your fist container:
```console
docker run -it --name [your_container_name] [your_image_name]
```
You should now have access to an interactive terminal within your container.  
On this terminal, open a Python console and check that Pytorch is installed.
```python
import torch
print(torch.__version__)
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

*   `colorize_app.py`
*   `mnist_app.py`
*   `mnist.pth`
*   `unet.pth`

Create a new container, this time mounting a shared volume with the following command:
```console
docker run -it --name [container_name] -v ~/[your_folder_path]:/workspace/[folder_name] [image_name]
```

Try to run one of your Gradio applications using the interactive mode.

```bash
cd [folder_name]
python colorize_app.py
```

Leave the container and look at your folder on your local machine. What can you see?

Now try to run your applications on your cloud instance.  
Send the `Dockerfile` and the folder containing your applications to your cloud instance.
On the cloud instance, build your image and run your container and your app in background mode.

```bash
sudo docker exec -t container1 python ./devlop/colorize_app.py --weights_path ./devlop/unet.pth
```

This is it for this session.  
Do not hesitate to play a little more with Docker.  
For instance try to train the MNIST classifier directly in your container and to collect the tensorboard logs and the resulting weights on your local machine.
