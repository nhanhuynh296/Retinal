'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import dataset

# dimensions of our images.
img_width, img_height = 1000, 1000  # todo increase to 2000
#
# train_data_dir = 'data/train'
# validation_data_dir = 'data/validation'
# nb_train_samples = 2000
# nb_validation_samples = 800
epochs = 50
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3,) + dataset.IMG_SIZE
else:
    input_shape = dataset.IMG_SIZE + (3,)

model = Sequential([tf.keras.layers.Rescaling(1. / 255, input_shape=dataset.IMG_SIZE + (3,))])
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


# def recall_m(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall = true_positives / (possible_positives + K.epsilon())
#     return recall
#
# def precision_m(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     return precision
#
# def f1_m(y_true, y_pred):
#     precision = precision_m(y_true, y_pred)
#     recall = recall_m(y_true, y_pred)
#     return 2*((precision*recall)/(precision+recall+K.epsilon()))
#
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy', recall_m, precision_m, f1_m])

# this is the augmentation configuration we will use for training
# train_datagen = ImageDataGenerator(rescale=1. / 255)

# test_datagen = ImageDataGenerator(rescale=1. / 255)  # rescale=1. / 255)

# train_generator = train_datagen.flow_from_directory(
#     train_data_dir,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode='binary')
#
# validation_generator = test_datagen.flow_from_directory(
#     validation_data_dir,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode='binary')

# steps_per_epoch the number of batch iterations before a training epoch is considered finished. If you have a
# training set of fixed size you can ignore it but it may be useful if you have a huge data set or if you are
# generating random data augmentations on the fly, i.e. if your training set has a (generated) infinite size. If you
# have the time to go through your whole training data set I recommend to skip this parameter. validation_steps
# similar to steps_per_epoch but on the validation data set instead on the training data. If you have the time to go
# through your whole validation data set I recommend to skip this parameter.

# steps_per_epoch=nb_train_samples // dataset.BATCH_SIZE,
# validation_steps=nb_validation_samples // dataset.BATCH_SIZE

# model.fit(
#     dataset.train_dataset,
#     epochs=epochs,
#     validation_data=dataset.validation_dataset,
#     )
# model.save('results/Building powerful image classification models using very little data/Building powerful image small'
#            'classification models using very little data small.h5')

