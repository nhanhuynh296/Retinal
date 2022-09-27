import cv2
import os

import helper.get_filenames
from helper import get_filenames
from ProcessImage import ProcessImage
from cv2 import imwrite
from dotenv import load_dotenv

load_dotenv("../.env")

CONFIG = "ONE"  # ONE       MULTI       CLASSES"
# if CONFIG is MULTI
CLASS = "GOOD"  # GOOD      BAD
folder_name = ""
IMAGE_TO_PROCESS = 0
########################################################################################################################
if CONFIG == "ONE":
    IMAGE_TO_PROCESS = IMAGE_TO_PROCESS or 1000
    file_dir, file_name = os.getenv('TEST_ONE'), "bad.png"
    if not file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        raise TypeError("Wrong image type extension")
    processed_image = ProcessImage(file_dir + file_name, "../output/").get_processed_image()
    path = os.path.join(os.getenv('OUT_DIR'), "ONE", folder_name)
    if not os.path.exists(path):
        os.mkdir(path)
    status = cv2.imwrite(f"{path}/{file_name}", processed_image)
    print("OK")
    while True:
        cv2.imshow(f"{os.getenv('OUT_DIR')}/ONE/{file_name}", processed_image)
        # set window on top
        cv2.setWindowProperty(f"{os.getenv('OUT_DIR')}/ONE/{file_name}", cv2.WND_PROP_TOPMOST, 1)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    if not status:
        raise TypeError(f"Image or path is possible None, try printing... {processed_image}")
elif CONFIG == "MULTI":
    IMAGE_TO_PROCESS = IMAGE_TO_PROCESS or 1000
    get_file = get_filenames.Get_filenames(os.getenv(f'IMG_TRAIN_{CLASS}'), IMAGE_TO_PROCESS)
    file_dirs = iter(get_file)
    path = os.path.join(os.getenv('OUT_DIR'), folder_name or f"{CLASS}_MULTI")
    if not os.path.exists(path):
        os.mkdir(path)
    for i, (file_dir, file_name) in enumerate(file_dirs):
        if None in [file_dir, file_name]:
            break
        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            continue
        processed_image = ProcessImage(file_dir + file_name, "").get_processed_image()
        status = imwrite(f"{path}/{file_name}", processed_image)
        if not status:
            raise TypeError(f"Image is possible None, try printing... {processed_image}")
        i += 1
        helper.get_filenames.printProgressBar(iteration=i, total=IMAGE_TO_PROCESS, length=100)
elif CONFIG == "CLASSES":
    IMAGE_TO_PROCESS = IMAGE_TO_PROCESS or 1000
    for dataset_type in ("TRAIN", "TEST"):
        for image_type in ("GOOD", "BAD"):
            path = os.path.join(os.getenv('OUT_DIR'), folder_name or "CLASS", f"{dataset_type}/{image_type}")
            os.mkdir(path)
            if not os.path.exists(path):
                os.mkdir(path)
            get_file = get_filenames.Get_filenames(os.getenv(f'IMG_{dataset_type}_{image_type}'), IMAGE_TO_PROCESS)
            file_dirs = iter(get_file)
            for i, (file_dir, file_name) in enumerate(file_dirs):
                if None in [file_dir, file_name]:
                    break
                # Check if file is image
                if not file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                    continue
                processed_image = ProcessImage(file_dir + file_name, "").get_processed_image()
                if processed_image.shape != (1000, 1000,3): continue
                status = imwrite(f"{path}/{file_name}", processed_image)
                if not status:
                    raise TypeError(f"Image is possible None, try printing... {processed_image}")
                i += 1
                helper.get_filenames.printProgressBar(prefix=f"{dataset_type, image_type}", iteration=i, total=IMAGE_TO_PROCESS, length=100)
quit(0)

