import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

INPUT_PATH = "/home/jupyter/DATA/"#"/kaggle/input/ranzcr-clip-catheter-line-classification/"
TRAIN_PATH=INPUT_PATH+"train/"
TEST_PATH=INPUT_PATH+"test/"

print(f"Number of train images {len(os.listdir(TRAIN_PATH))}\nNumber of test images {len(os.listdir(TEST_PATH))}")

ANNOTATIONS=INPUT_PATH+"train.csv"


targets=["ETT - Abnormal","ETT - Borderline","ETT - Normal",
         "NGT - Abnormal","NGT - Borderline","NGT - Incompletely Imaged","NGT - Normal",
         "CVC - Abnormal","CVC - Borderline","CVC - Normal",
         "Swan Ganz Catheter Present"]


import tensorflow as tf
print(f"Tensorflow version: {tf.__version__}")

print(tf.config.list_physical_devices(
    device_type=None))

# Define the strategy

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    print("Running on TPU ", tpu.cluster_spec().as_dict()["worker"])
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    
except ValueError:
    print("Not connected to a TPU runtime. Using CPU/GPU strategy")
    strategy = tf.distribute.MirroredStrategy()
    
from tensorflow.keras.applications import EfficientNetB7,EfficientNetB5,EfficientNetB3,EfficientNetB0

# Transfer learning using efficientnet

IMG_SIZE=512
NUM_CLASSES=len(targets)
efficientnet=EfficientNetB5 #ResNet152


with strategy.scope():
    model = tf.keras.Sequential([
              tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
              efficientnet(include_top=False, weights='imagenet',),
              tf.keras.layers.Flatten(),
              tf.keras.layers.Dense(512,activation='relu'),
              tf.keras.layers.Dense(NUM_CLASSES,activation="sigmoid"), # multi-label
    ])
    
    #model.compile(
    #    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC(multi_label=True)])
    

    
print(model.summary())


from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True)
from IPython.display import Image
Image(filename='model.png') 


#
#for layer in model.layers:
#    if layer.name == "efficientnetb3":
#        layer.trainable=False
        
        
#from kaggle_datasets import KaggleDatasets
PATH_GCS = INPUT_PATH#KaggleDatasets().get_gcs_path("ranzcr-clip-catheter-line-classification")

PATH_GCS_TRAINING=PATH_GCS+"/train_tfrecords/"+"*.tfrec"
filenames = tf.io.gfile.glob(PATH_GCS_TRAINING)
raw_dataset = tf.data.TFRecordDataset(filenames)


# Inspect a raw tfrecord
#for raw_record in raw_dataset.take(1):
#      print(repr(raw_record))


# https://www.tensorflow.org/tutorials/load_data/tfrecord#reading_a_tfrecord_file
# https://www.kaggle.com/venkat555/ranzcr-clip-tpu-densenet-with-kfold

feature_description = {
    "StudyInstanceUID"           : tf.io.FixedLenFeature([], tf.string),
    "image"                      : tf.io.FixedLenFeature([], tf.string),
    "ETT - Abnormal"             : tf.io.FixedLenFeature([], tf.int64), 
    "ETT - Borderline"           : tf.io.FixedLenFeature([], tf.int64), 
    "ETT - Normal"               : tf.io.FixedLenFeature([], tf.int64), 
    "NGT - Abnormal"             : tf.io.FixedLenFeature([], tf.int64), 
    "NGT - Borderline"           : tf.io.FixedLenFeature([], tf.int64), 
    "NGT - Incompletely Imaged"  : tf.io.FixedLenFeature([], tf.int64), 
    "NGT - Normal"               : tf.io.FixedLenFeature([], tf.int64), 
    "CVC - Abnormal"             : tf.io.FixedLenFeature([], tf.int64), 
    "CVC - Borderline"           : tf.io.FixedLenFeature([], tf.int64), 
    "CVC - Normal"               : tf.io.FixedLenFeature([], tf.int64), 
    "Swan Ganz Catheter Present" : tf.io.FixedLenFeature([], tf.int64),
}

def _parse_function(example_proto):
  # Parse the input `tf.train.Example` proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, feature_description)


def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [1024,1024, 3])
    return image

def input_fc(example_proto):
    example = _parse_function(example_proto)
    image = decode_image(example['image']) 
    image = tf.image.resize(image, [IMG_SIZE,IMG_SIZE])
    
    label=[tf.cast(example[lbl],tf.float32) for lbl in targets]
    #label=tf.one_hot(label, len(targets))
    
    return image,label


# Define a basic image augmentation with tensoflow

def augmentation_fc(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.5)
    image = tf.image.random_saturation(image, 0, 2)
    image = tf.image.adjust_saturation(image, 3)
    return image, label


def get_dataset(filenames,shuffle,batch_size,mode="train",augmentation=True):
    AUTOTUNE=tf.data.experimental.AUTOTUNE
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    dataset = dataset.map(input_fc, num_parallel_calls=AUTOTUNE)
    if mode=="train": 
        if augmentation: dataset = dataset.map(augmentation_fc, num_parallel_calls=AUTOTUNE)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(shuffle, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


for image,label in get_dataset(filenames,shuffle=1,batch_size=1).take(1):
    plt.figure()
    plt.imshow(image.numpy()[0])
    print(image.numpy().shape, label.numpy().shape)
    print(label.numpy())
    
    
## Train - validation split and input functions
len(filenames)
filenames_train = filenames[:11]
filenames_val = filenames[11:]

BATCH_SIZE = 128
SHUFFLE = 3


import re
# From https://www.kaggle.com/venkat555/ranzcr-clip-tpu-densenet-with-kfold
def count_data_items(filenames):
    #the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)
    #c = 0
    #for filename in filenames:
    #    c += sum(1 for _ in tf.data.TFRecordDataset(filename))
    #return c
NUM_TRAINING_IMAGES = count_data_items(filenames_train)
NUM_TEST_IMAGES = count_data_items(filenames_val)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

def get_train_dataset():
    return get_dataset(filenames_train,BATCH_SIZE,SHUFFLE,augmentation=True)


def get_val_dataset():
    return get_dataset(filenames_val,BATCH_SIZE,SHUFFLE,mode="val")


EPOCHS=2000

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'model.h5', save_best_only=True, monitor='val_auc', mode='max')
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                  min_delta=0, 
                                                  patience=100, verbose=0, 
                                                  restore_best_weights=True)
tensorboard_callback = tf.keras.callbacks.TensorBoard("logs")    

try: #https://www.tensorflow.org/guide/keras/save_and_serialize#saving_loading_only_the_models_weights_values
    model.load_weights('/home/jupyter/RANZCR-kaggle-image-classification/model.h5') #model.load_weights(checkpoint_filepath)
    print("Weights loaded")
except:
    print("No compatible weights found!")


hist = model.fit(get_train_dataset(), epochs=EPOCHS,steps_per_epoch=STEPS_PER_EPOCH,
                 validation_data=get_val_dataset(),
                 callbacks=[checkpoint, early_stopping, tensorboard_callback],
                 )