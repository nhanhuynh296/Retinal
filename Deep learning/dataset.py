import os
import tensorflow
from dotenv import load_dotenv

load_dotenv("../.env")

BATCH_SIZE = 32
IMG_SIZE = (512, 512)


#
# _train_ds = tensorflow.keras.utils.image_dataset_from_directory(os.getenv('IMG_TRAIN'),
#                                                        # validation_split=0.2,
#                                                        # subset="training",
#                                                        seed=123,
#                                                        image_size=IMG_SIZE,
#                                                        batch_size=BATCH_SIZE)
_test_ds = tensorflow.keras.utils.image_dataset_from_directory(os.getenv('IMG_TEST'),
                                                      # validation_split=0.2,
                                                      # subset="validation",
                                                      seed=123,
                                                      image_size=IMG_SIZE,
                                                      batch_size=BATCH_SIZE)
# Use buffered prefetching to load images from disk without having I/O become blocking
AUTOTUNE = tensorflow.data.AUTOTUNE
# train_dataset = _train_ds.prefetch(buffer_size=AUTOTUNE)
validation_dataset = _test_ds.prefetch(buffer_size=AUTOTUNE)

