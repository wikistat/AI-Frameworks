# Librairies
print("Load Libraries")
import time
import pickle
import hashlib
import os
import tensorflow.keras.preprocessing.image as kpi
import tensorflow.keras.layers as kl
import tensorflow.keras.models as km

from tensorflow.python.client import device_lib

MODE = "GPU" if "GPU" in [k.device_type for k in device_lib.list_local_devices()] else "CPU"
print(MODE)

## Argument
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)

parser.add_argument('--data_dir', type=str,
                    default="/Users/bguillouet/Insa/TP_Insa/dev/IA-Frameworks/tp_google_cloud/data")
parser.add_argument('--results_dir', type=str,
                    default="/Users/bguillouet/Insa/TP_Insa/dev/IA-Frameworks/tp_google_cloud/results")
parser.add_argument('--model_dir', type=str,
                    default="/Users/bguillouet/Insa/TP_Insa/dev/IA-Frameworks/tp_google_cloud/model")

args = parser.parse_args()

## Definition des variables

img_width = 150
img_height = 150

N_train = len(os.listdir(args.data_dir+"/train/cats/")) + len(os.listdir(args.data_dir+"/train/dogs/"))
N_val = len(os.listdir(args.data_dir+"/validation/cats/")) + len(os.listdir(args.data_dir+"/validation/dogs/"))
print("%d   %d"%(N_train, N_val))

## Data Generator

train_datagen = kpi.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    args.data_dir + "/train/",  # this is the target directory
    target_size=(img_width, img_height),
    batch_size=args.batch_size,
    class_mode='binary')

valid_datagen = kpi.ImageDataGenerator(rescale=1. / 255)

validation_generator = valid_datagen.flow_from_directory(
    args.data_dir + "/validation/",
    target_size=(img_width, img_height),
    batch_size=args.batch_size,
    class_mode='binary')

## Definition du modEle

model_conv = km.Sequential()
model_conv.add(kl.Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3), data_format="channels_last"))
model_conv.add(kl.Activation('relu'))
model_conv.add(kl.MaxPooling2D(pool_size=(2, 2)))

model_conv.add(kl.Conv2D(32, (3, 3)))
model_conv.add(kl.Activation('relu'))
model_conv.add(kl.MaxPooling2D(pool_size=(2, 2)))

model_conv.add(kl.Conv2D(64, (3, 3)))
model_conv.add(kl.Activation('relu'))
model_conv.add(kl.MaxPooling2D(pool_size=(2, 2)))

model_conv.add(kl.Flatten())  # this converts our 3D feature maps to 1D feature vectors
model_conv.add(kl.Dense(64))
model_conv.add(kl.Activation('relu'))
model_conv.add(kl.Dropout(0.5))
model_conv.add(kl.Dense(1))
model_conv.add(kl.Activation('sigmoid'))

model_conv.compile(loss='binary_crossentropy',
                   optimizer='rmsprop',
                   metrics=['accuracy'])

## Learning

print("Start Learning")
ts = time.time()
model_conv.fit_generator(train_generator, steps_per_epoch=N_train // args.batch_size, epochs=args.epochs,
                         validation_data=validation_generator, validation_steps=N_val // args.batch_size)
te = time.time()
t_learning = te - ts

## Test

print("Start predicting")
ts = time.time()
score_train = model_conv.evaluate_generator(train_generator, N_train / args.batch_size, verbose=1)
score_val = model_conv.evaluate_generator(validation_generator, N_val / args.batch_size, verbose=1)
te = time.time()
t_prediction = te - ts

## Save Model

### Creer un identifiant unique a partir des parametres du script
args_str = "_".join([k + ":" + str(v) for k, v in sorted(vars(args).items(), key=lambda x : x[0])])
id_str = hashlib.md5(args_str.encode("utf8")).hexdigest()
print("AAAAAAAAAAA")
print(args_str)
print(id_str)
model_conv.save(args.model_dir + "/" + id_str + ".h5")

## Save results


print("Save results")
results = vars(args)
results.update({"t_learning": t_learning, "t_prediction": t_prediction, "accuracy_train": score_train,
                 "accuracy_val": score_val})

print(results)
pickle.dump(results, open(args.results_dir + "/" + id_str + ".pkl", "wb"))
