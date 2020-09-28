# Librairies
print("Load Libraries")
import time
import pickle
import os
import tensorflow.keras.preprocessing.image as kpi
import tensorflow.keras.layers as kl
import tensorflow.keras.models as km

from tensorflow.python.client import device_lib

MODE = "GPU" if "GPU" in [k.device_type for k in device_lib.list_local_devices()] else "CPU"
print(MODE)

## Argument
import argparse

# TODO Write here the parameters that can be given as inputs to the algorithm.
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default=)
args = parser.parse_args()


## Data Generator

img_width = 150
img_height = 150

N_train = len(os.listdir(args.data_dir+"/train/cats/")) + len(os.listdir(args.data_dir+"/train/dogs/"))
N_val = len(os.listdir(args.data_dir+"/validation/cats/")) + len(os.listdir(args.data_dir+"/validation/dogs/"))
print("%d   %d"%(N_train, N_val))

# TODO Write here code to generate obh train and validation generator
train_datagen = kpi.ImageDataGenerator()

train_generator = train_datagen.flow_from_directory()

valid_datagen = kpi.ImageDataGenerator()

validation_generator = valid_datagen.flow_from_directory()

## Model
# TODO Write a simple convolutional neural network


model_conv =

## Learning

print("Start Learning")
ts = time.time()
history = model_conv.fit_generator(train_generator, steps_per_epoch=N_train // args.batch_size, epochs=args.epochs,
                         validation_data=validation_generator, validation_steps=N_val // args.batch_size)
te = time.time()
t_learning = te - ts

## Test
# TODO Calculez l'accuracy de votre modèle sur le jeu d'apprentissage et sur le jeu de validation.

print("Start predicting")
ts = time.time()
score_train =
score_val =
te = time.time()
t_prediction = te - ts



args_str = "epochs_%d_batch_size_%d" %(args.epochs, args.batch_size)

## Save Model

# TODO Save model in model folder


## Save results

## TODO Save results (learning time, prediction time, train and test accuracy, history.history object) in result folder




















