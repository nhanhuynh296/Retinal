import numpy
import tensorflow as tf
from tensorflow.python.keras import backend as K
import dataset
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tqdm


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


model_name = "model_full"

model_names = ["Building powerful image classification models using very little data",
               "Building powerful image classification models using very little data_50ep", ]
# "Image Classification from scratch",
# "Transfer Learning TF"]

print(model_name)
model = tf.keras.models.load_model(filepath=f'results/{model_name}.h5',
                                   custom_objects={"recall_m": recall_m, "precision_m": precision_m,
                                                   "f1_m": f1_m}, compile=False)

dataset_unbatched = dataset.validation_dataset  # unbatch()
pred = []
labels = []
pred_AUC = []

# 3369
for (image_batch, label_batch) in tqdm.tqdm(dataset_unbatched, total=dataset_unbatched.cardinality().numpy()):
    # helper.get_filenames.printProgressBar(i, 3369)
    # img_array = tf.keras.preprocessing.image.img_to_array(image)
    # img_array = tf.expand_dims(img_array, 0)  # Create batch size axis
    prediction_batch = model.predict(image_batch, verbose=0)
    for prediction in prediction_batch:
        pred_AUC.append(prediction[0])
        pred.append(prediction[0] > 0.5)
    labels.extend((label_batch.numpy() == 1).astype(int))  # Bad: 0, Good: 1, Outlier: 2

# if len(labels) == len(pred):
print(classification_report(y_true=labels, y_pred=pred))
print(confusion_matrix(y_true=labels, y_pred=pred))
roc_auc_score(y_true=labels, y_score=pred)
