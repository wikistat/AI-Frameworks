# Librairies
print("Load Libraries")
import os

import numpy as np
import pandas as pd

import tensorflow.keras.preprocessing.image as kpi
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



# TODO Define generator.


## Data Generator
img_width = 150
img_height = 150


data_dir_test = args.data_dir+'/test'
N_test = len(os.listdir(data_dir_test+"/test"))

test_datagen = kpi.ImageDataGenerator()
test_generator = test_datagen.flow_from_directory()


## Download model
# Todo Download model saved in learning script.
args_str = "epochs_%d_batch_size_%d" %(args.epochs, args.batch_size)


## Prediction
# Todo Generate prediction.

## Save prediction in csv
# TODO Save the results in a csv file.









