import cv2
import numpy as np
from os import walk

# IMG_PATH = "/Users/nathan/Desktop/COSC428/retinal/data/"
IMG_PATH = "../data/"

image_name = "test.jpg"


def binary_threshold(image):
    ret, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    return image


def resize(image, size):
    image = cv2.resize(image, (int(image.shape[1] * size), int(image.shape[0] * size)))
    return image


def morphological(image):
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.morphologyEx(src=image, op=cv2.MORPH_OPEN, kernel=kernel)
    image = cv2.morphologyEx(src=image, op=cv2.MORPH_CLOSE, kernel=kernel)
    image = cv2.erode(src=image, kernel=kernel)
    return image


def blur_and_grayscale(img, blur_kernel=9):
    blur = cv2.GaussianBlur(img, (blur_kernel, blur_kernel), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    return gray


def hough_circle_trackbar_setup(window_name):
    def nothing(x):
        """Callback event"""
        pass

    cv2.createTrackbar('Canny Threshold', window_name, 1, 500, nothing)
    cv2.createTrackbar('Accumulator Threshold', window_name, 1, 500, nothing)
    cv2.createTrackbar("Min Radius", window_name, 0, 100, nothing)
    #   Set some default parameters
    cv2.setTrackbarPos("Canny Threshold", window_name, 100)
    cv2.setTrackbarPos("Accumulator Threshold", window_name, 80)


def get_hough_circle_trackbar_pos(window_name):
    canny = cv2.getTrackbarPos('Canny Threshold', window_name)
    transform = cv2.getTrackbarPos('Accumulator Threshold', window_name)
    min_radius = cv2.getTrackbarPos('Min Radius', window_name)
    return canny, transform, min_radius


def get_hough_circle(gray_image, window_name):
    # canny, transform, min_radius = get_hough_circle_trackbar_pos(window_name)
    canny = 100
    transform = 80
    min_radius = 0
    circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, 1, 20, param1=canny, param2=transform,
                               minRadius=min_radius,
                               maxRadius=None)
    return np.uint16(np.around(circles))


def draw_hough_circles(image, circles):
    circles = np.uint16(np.around(circles))
    if circles is not None:
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
    return image


def crop_from_circle(grey_image, original, circles):
    height, width = grey_image.shape
    mask = np.zeros((height, width), np.uint8)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        circle_mask = cv2.circle(mask, (i[0], i[1]), i[2] + 20, (255, 255, 255), thickness=-1)
    masked_data = cv2.bitwise_and(original, original, mask=circle_mask)
    # draw_hough_circles(masked_data, circles)  # scale dynamically
    _, thresh = cv2.threshold(circle_mask, 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    x, y, w, h = cv2.boundingRect(contours[0])
    crop = masked_data[y:y + h, x:x + w]
    masking = thresh[y:y + h, x:x + w]
    return crop, masking


def errosion_circular(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    image = cv2.erode(image, kernel, iterations=1)
    return image


def clahe_enhancement(image):
    """Applied to red and green band"""
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    image[:, :, 1] = clahe.apply(image[:, :, 1])
    image[:, :, 2] = clahe.apply(image[:, :, 2])
    return image


# while True:
#     image_name = "test.jpg"  # todo next(file_names)
#     img = cv2.imread(IMG_PATH + image_name)

# thresholding

#
# # img = cv2.addWeighted(img, 0.5, threshold, 0.5, 0) #todo
# # morphology
# kernel = np.ones((3, 3), np.uint8)
# for i in range(1):
#     img = cv2.morphologyEx(src=img, op=cv2.MORPH_OPEN, kernel=kernel)
#     img = cv2.morphologyEx(src=img, op=cv2.MORPH_CLOSE, kernel=kernel)
#     img = cv2.erode(img, kernel, iterations=1)
#
# # blur layer
# blur = cv2.GaussianBlur(img, (9, 9), 0)
# gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
#
# # hough circle
#
# canny = cv2.getTrackbarPos('Canny Threshold', image_name)
# transform = cv2.getTrackbarPos('Accumulator Threshold', image_name)
# minRadius = cv2.getTrackbarPos('Min Radius', image_name)
# maxRadius = cv2.getTrackbarPos('Max Radius', image_name)
# circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=canny, param2=transform, minRadius=minRadius,
#                            maxRadius=None)

# show and wait, space to next img or q to quit
# show_and_wait_key(file_name=image_name, matrix=img)


def main():
    # BGR
    window_name = image_name
    cv2.namedWindow(image_name)
    img = cv2.imread(IMG_PATH + image_name)
    img = resize(img, 0.3)
    original = img.copy()
    img[:, :, 0] = 0
    img[:, :, 1] = 0
    img = binary_threshold(img)
    img = morphological(img)
    # hough_circle_trackbar_setup(window_name)
    gray = blur_and_grayscale(img)
    circles = get_hough_circle(gray_image=gray, window_name=window_name)

    cropped, mask = crop_from_circle(gray, original, circles)
    mask = errosion_circular(mask)
    cropped = cv2.bitwise_and(cropped, cropped, mask=mask)
    # test_trackbar(window_name=window_name)
    img = clahe_enhancement(cropped)
    img = blur_and_grayscale(img, blur_kernel=5)

    while True:
        # clahe_pos = cv2.getTrackbarPos('clahe', window_name)
        # size = cv2.getTrackbarPos('win_size', window_name)

        cv2.imshow(f'{image_name}', img)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


def test_trackbar(window_name):
    def nothing(x):
        pass

    cv2.createTrackbar("clahe", window_name, 0, 10, nothing)
    cv2.createTrackbar("win_size", window_name, 1, 100, nothing)
    cv2.setTrackbarPos("clahe", window_name, 1)
    cv2.setTrackbarPos("win_size", window_name, 1)


if __name__ == "__main__":
    main()
