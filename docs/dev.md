# Development for Data Scientist:
(Python environment, Github Repo and Python Script)
---------------------------------------------------

<iframe width="560" height="315" src="https://www.youtube.com/embed/gZLeTHQzloE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

*   [Slides](https://github.com/wikistat/AI-Frameworks/raw/master/slides/Code_Development_Python.pdf)
*   [Practical session](https://github.com/wikistat/AI-Frameworks/blob/master/CodeDevelopment/TP.pdf)

The goal of this lab is to create a python script in which we will train a model on a subset of [Imagenet](https://image-net.org/) called [Imagenette](https://github.com/fastai/imagenette) composed of 10 easily classified classes (tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute).
The training hyper-parameters will be provided by the user during execution.
We will also use Tensorboard to monitor its learning during script execution. 

Let's first begin with downloading and extracting the dataset.  
If you are on Linux, use the following command to download the dataset: 
```
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
```
and extract it using this command:
```
tar zxvf imagenette2.tgz
```
If you are on Windows, download the dataset from this [link](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz) and extract it using [7Zip](https://www.7-zip.org/download.html)


Use you favorite coding editor to create a new file `transfer_learning.py`.

# TODO rajouter le argparse

To avoid overloading our RAM, we will use two data generators to load our dataset on the fly:

```python
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

base_dir = 'imagenette2'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')
```

Since you are probably working on a computer that does not have a graphics card, we will use a pretrained [VGG16](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiCt_H21PzxAhUkxIUKHbVfDf4QFjAAegQIBBAD&url=https%3A%2F%2Farxiv.org%2Fabs%2F1409.1556&usg=AOvVaw17ak86ejVzNlyA2N-WpWmZ) as a backbone of our neural architecture on wich we will plug a simple linear classier.

Add the following import on top of your file:
```python
from tensorflow.keras.applications import VGG16
```
The following code will load a pre-trained VGG16 without its fnal layers used for classification:
```python
from tensorflow.keras.applications import VGG16

backbone = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))
```
