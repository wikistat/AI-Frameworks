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
# TODO Les paramètres de l'environnement sont défini ici. Ajoutez les directions de vos dossier en local par défault.


parser.add_argument('--data_dir', type=str, default="")
parser.add_argument('--results_dir', type=str,  default="")
parser.add_argument('--model_dir', type=str, default="")

# TODO ajoutez ici les différents paramètres du modèle que vous souhaitez pour manipuler en entré de script
parser.add_argument('--', type=str, default="")

args = parser.parse_args()

## Definition des variables

img_width = 150
img_height = 150

## Data Generator

# TODO définissez ici le différents "generator" qui vous permettra de lire les données et généraler les batchs
N_test =

test_datagen = kpi.ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(e)


## Telechargement du modele
args_str = "_".join([k + ":" + str(v) for k, v in sorted(vars(args).items(), key=lambda x : x[0])])
id_str = hashlib.md5(args_str.encode("utf8")).hexdigest()
model_conv = km.load_model(args.model_dir + "/" + id_str + ".h5")



## Prediction
# TODO Effectuez la prédiction

test_prediction = model_conv.predict_generator()

## Save prediction in csv
# TODO Sauvegarder vos résultats dans le dossier results_dir sous forme d'un fichier csv.
