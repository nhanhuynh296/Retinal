
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

train_data_gen = ImageDataGenerator()
train_data = train_data_gen.flow_from_directory(
    directory="/media/nathan/Data/DiabeticRetinopathyScreeningImageDatabase/chen/RetinalImageQualityAssessment/data/train",
    target_size=(2000, 2000))
test_data_gen = ImageDataGenerator()
test_data = test_data_gen.flow_from_directory(
    directory="/media/nathan/Data/DiabeticRetinopathyScreeningImageDatabase/chen/RetinalImageQualityAssessment/data/test",
    target_size=(2000, 2000))

from tensorflow.keras.applications.inception_v3 import InceptionV3

VGGmodel = InceptionV3(weights='imagenet', include_top=True)

for layers in VGGmodel.layers[:19]:
    layers.trainable = False

"""Since my problem is to detect cats and dogs and it has two classes so the last dense layer of my model should be a 2 
unit softmax dense layer. Here I am taking the second last layer of the model which is dense layer with 4096 units 
and adding a dense softmax layer of 2 units in the end. In this way I will remove the last layer of the VGG16 model 
which is made to predict 1000 classes. """
X = VGGmodel.layers[-2].output
predictions = Dense(2, activation="softmax")(X)
model_final = Model(VGGmodel.input, predictions)
model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
                    metrics=["accuracy"])

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False,
                             mode='auto', save_freq='epoch')
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=40, verbose=1, mode='auto')
model_final.fit(train_data, steps_per_epoch=2, epochs=100, validation_data=test_data,
                validation_steps=1, callbacks=[checkpoint, early])
model_final.save_weights("vgg16_1.h5")
