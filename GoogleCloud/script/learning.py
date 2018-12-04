# Librairies
print("Load Libraries")

import time
import pickle
import hashlib
import argparse
import os
import keras.preprocessing.image as kpi
import keras.layers as kl
import keras.models as km

from tensorflow.python.client import device_lib

MODE = "GPU" if "GPU" in [k.device_type for k in device_lib.list_local_devices()] else "CPU"
print(MODE)

## Argument



parser = argparse.ArgumentParser()

# TODO Les paramètres de l'environnement sont défini ici. Ajoutez les directions de vos dossier en local par défault.

parser.add_argument('--data_dir', type=str, default="")
parser.add_argument('--model_dir', type=str, default="")
parser.add_argument('--results_dir', type=str, default="")

# TODO ajoutez ici les différents paramètres du modèle que vous souhaitez pour manipuler en entré de script

parser.add_argument('--', type=str, default="")




args = parser.parse_args()

## Définition des variables

img_width = 150
img_height = 150
# TODO Stockez dans des variable 'N_train' et 'N_val' les tailles des echantillons d'apprentissage et de validation de l'échantillon
N_train =
N_val =



## Data Generator

# TODO définissez ici les différents "generator" qui vous permettront de lire les données et généraler les batchs

train_datagen = kpi.ImageDataGenerator()

train_generator = train_datagen.flow_from_directory()

valid_datagen = kpi.ImageDataGenerator()

validation_generator = valid_datagen.flow_from_directory()


## Définition du modèle

# TODO Définissez un premier modèle simple à l'aide de Keras

model_conv = km.Sequential()


model_conv.compile(loss=,
                   optimizer=,
                   metrics=)


## Learning

print("Start Learning")
ts = time.time()
model_conv.fit_generator(train_generator,± steps_per_epoch=N_train // args.batch_size, epochs=args.epochs,
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


## Save Model

### Créer un identifiant unique à partir des paramètres du script
args_str = "_".join([k + ":" + str(v) for k, v in sorted(vars(args).items(), key=lambda x : x[0])])
id_str = hashlib.md5(args_str.encode("utf8")).hexdigest()

# TODO Sauvez le modèle dans le dossier model_dir


## TODO   Stockez les resultat dans des variables ainsi que le temps d'execution de ces opérations dans le dossier results_dir


print("Save results")
results = vars(args)
results.update({"t_learning": t_learning, "t_prediction": t_prediction, "accuracy_train": score_train,
                 "accuracy_val": score_val})

print(results)
pickle.dump(results, open(args.results_dir + "/" + id_str + ".pkl", "wb"))

