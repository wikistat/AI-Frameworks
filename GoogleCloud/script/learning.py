# Librairies
print("Load Libraries")

import time
import pickle
import hashlib
import argparse


import keras.preprocessing.image as kpi
import keras.layers as kl
import keras.models as km

from tensorflow.python.client import device_lib

MODE = "GPU" if "GPU" in [k.device_type for k in device_lib.list_local_devices()] else "CPU"
print(MODE)

## Argument

# TODO ajouter ici les différents arguments que vous manipulerez lors de vos différents tests


parser = argparse.ArgumentParser()


parser.add_argument('--data_dir', type=str, default="")
parser.add_argument('--model_dir', type=str, default="")
parser.add_argument('--metadata_dir', type=str, default="")


args = parser.parse_args()

## Définition des variables

img_width = 150
img_height = 150

## Data Generator

# TODO définissez ici les différents "generator" qui vous permettront de lire les données

train_datagen = kpi.ImageDataGenerator()

train_generator = train_datagen.flow_from_directory()

valid_datagen = kpi.ImageDataGenerator()

validation_generator = valid_datagen.flow_from_directory()


## Définition du modèle

# TODO Définit un premier modèle simple à l'aide de Keras

model_conv = km.Sequential()


model_conv.compile(loss=,
                   optimizer=,
                   metrics=)


## Learning

# TODO


## Test

# TODO Définit un premier modèle simple à l'aide de Keras


## Save Model

### Créer un identifiant unique à partir des paramètres du script
args_str = "_".join([k + ":" + str(v) for k, v in vars(args).items()])
id_str = hashlib.md5(args_str.encode("utf8")).hexdigest()

# TODO

## Save Model

# TODO

model_conv.save(args.model_dir + "/" + id_str + ".h5")


## Save Metadata

print("Save Metadata")
metadata = vars(args)
metadata.update({"t_learning": t_learning, "t_prediction": t_prediction, "accuracy_train": score_train,
                 "accuracy_val": score_val})

print(metadata)
pickle.dump(metadata, open(args.metadata_dir + "/" + id_str + ".pkl", "wb"))

# TODO
