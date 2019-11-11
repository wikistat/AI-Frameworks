# Librairies
print("Load Libraries")
import os
import hashlib

import numpy as np
import pandas as pd

import tensorflow.keras.preprocessing.image as kpi
import tensorflow.keras.models as km


from tensorflow.python.client import device_lib
MODE = "GPU" if "GPU" in [k.device_type for k in device_lib.list_local_devices()] else "CPU"
print(MODE)

## Argument
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)

DATA_DIR = ""
parser.add_argument('--data_dir', type=str,
                    default=DATA_DIR+"/data/sample_2/")
parser.add_argument('--results_dir', type=str,
                    default=DATA_DIR+"/results/")
parser.add_argument('--model_dir', type=str,
                    default=DATA_DIR+"/model/")

args = parser.parse_args()

## Definition des variables

img_width = 150
img_height = 150

## Data Generator

data_dir_test = args.data_dir+'/test'
N_test = len(os.listdir(data_dir_test+"/test"))

test_datagen = kpi.ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    data_dir_test,
    target_size=(img_height, img_width),
    batch_size=args.batch_size,
    class_mode=None,
    shuffle=False)

## Telechargement du modele
args_str = "_".join([k + ":" + str(v) for k, v in sorted(vars(args).items(), key=lambda x : x[0])])
id_str = hashlib.md5(args_str.encode("utf8")).hexdigest()
model_conv = km.load_model(args.model_dir + "/" + id_str + ".h5")



## Prediction

test_prediction = model_conv.predict_generator(test_generator, N_test // args.batch_size, verbose=1)

## Save prediction in csv

images_test = test_generator.filenames
classes = [int(t>0.5) for t in test_prediction]

array = np.vstack((images_test, test_prediction[:,0], classes)).T
df = pd.DataFrame(array, columns=["filename","probabilities","classes"])
df.to_csv(args.results_dir+"/prediction_"+id_str+".csv", index=False)
