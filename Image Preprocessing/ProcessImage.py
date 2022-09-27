import cv2
import numpy as np
import math


def get_color_channel(img, channel):
    index = {"b": 0, "g": 1, "r": 2}
    # ret = np.zeros(img.shape)
    # ret[:,:,index[channel]] = img[:, :, index[channel]]
    return img[:, :, index[channel]]


def resize(image, to_size_or_percent):
    if isinstance(to_size_or_percent, float):
        height, width, *dimension = image.shape
        dimension = int(width * to_size_or_percent), int(height * to_size_or_percent)
    elif isinstance(to_size_or_percent, (tuple, list)):
        dimension = to_size_or_percent
    return cv2.resize(image, dimension, interpolation=cv2.INTER_AREA)


def binary_threshold(image, threshold=35):
    ret, image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return image


def morphological(image):
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.morphologyEx(src=image, op=cv2.MORPH_OPEN, kernel=kernel)
    image = cv2.morphologyEx(src=image, op=cv2.MORPH_CLOSE, kernel=kernel)
    image = cv2.erode(src=image, kernel=kernel)
    return image


def blur(img, blur_kernel=3):
    return cv2.GaussianBlur(img, (blur_kernel, blur_kernel), 0)


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def crop_info_from_bounding_box_with_pad(threshold_img, image):
    contours, _ = cv2.findContours(threshold_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_contours_area = 0
    pad = 5
    x_max, y_max, w_max, h_max = 0, 0, image.shape[1], image.shape[0]  # default case
    for cntr in contours:
        if cv2.contourArea(cntr) > max_contours_area:
            max_contours_area = cv2.contourArea(cntr)
            x_max, y_max, w_max, h_max = cv2.boundingRect(cntr)
        x, y, w, h = cv2.boundingRect(cntr)
        cv2.rectangle(image, (x_max - pad, y_max - pad), (x_max + w_max + 2 * pad, y + h_max + 2 * pad), (255, 0, 0),
                      1)  # print the bounding box

    # check so that if it can not draw bounding rectangle then it not overflow outside
    x_max = max(0, x_max - pad)
    y_max = max(0, y_max - pad)
    w_max = min(image.shape[1] - x_max, w_max + 2 * pad)
    h_max = min(image.shape[0] - y_max, h_max + 2 * pad)
    return x_max, y_max, w_max, h_max


def crop_image(image, crop_info):
    x, y, w, h = crop_info
    image = image[y:y + h, x:x + w]
    return image


def errosion_circular(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    image = cv2.erode(image, kernel, iterations=1)
    return image


def clahe_enhancement(image):
    """Applied to red and green band"""
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(10, 10))
    image = clahe.apply(image)
    return image


def equalize_illumination(img, window_size, avg_intensity):
    image_shape = img.shape
    half = int(window_size / 2)
    equalized_image = img.copy()
    for y in range(image_shape[1]):
        for x in range(image_shape[0]):
            # check pixel is not black
            if img[x, y] == 0:
                continue
            if x < half:
                if y < half:
                    mean = np.mean(img[0:x + half, 0:y + half]
                                   [img[0:x + half, 0:y + half] != 0])
                else:
                    mean = np.mean(img[0:x + half, y - half:y + half]
                                   [img[0:x + half, y - half:y + half] != 0])
            else:
                if y < half:
                    mean = np.mean(img[x - half:x + half, 0:y + half]
                                   [img[x - half:x + half, 0:y + half] != 0])
                else:
                    mean = np.mean(img[x - half:x + half, y - half:y + half]
                                   [img[x - half:x + half, y - half:y + half] != 0])
            new_val = img[x, y] + avg_intensity - mean
            equalized_image[x, y] = max(0, new_val)
    return equalized_image


def gamma_correction(src, gamma):
    """
    Formula found in
    https://lindevs.com/apply-gamma-correction-to-an-image-using-opencv/
    """
    invert_gammma = 1 / gamma

    table = [((i / 255) ** invert_gammma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)


def build_filters(sigma=2, YLength=9):
    """Builds a set of 12 filters that model vessels in all orientations from
    0 to 180 degrees with an angular resolution of 15 degrees.

    Returns list of generated filters.

    Refer to section 4D of readme reference 2 for more information.

    Function sourced from:
    https://www.programmersought.com/article/64024088316/"""

    filters = []
    width_of_the_kernel = np.ceil(np.sqrt((6 * np.ceil(sigma) + 1) ** 2 + YLength ** 2))
    if np.mod(width_of_the_kernel, 2) == 0:
        width_of_the_kernel = width_of_the_kernel + 1
    width_of_the_kernel = int(round(width_of_the_kernel))
    for theta in np.arange(15, 195, 180 / 12):
        matchFilterKernel = np.zeros((width_of_the_kernel, width_of_the_kernel),
                                     dtype=np.float32)
        for x in range(width_of_the_kernel):
            for y in range(width_of_the_kernel):
                half_length = (width_of_the_kernel - 1) / 2
                x_ = (x - half_length) * np.cos(theta) + (y - half_length) * np.sin(theta)
                y_ = -(x - half_length) * np.sin(theta) + (y - half_length) * np.cos(theta)
                if abs(x_) > 3 * np.ceil(sigma):
                    matchFilterKernel[x][y] = 0
                elif abs(y_) > (YLength - 1) / 2:
                    matchFilterKernel[x][y] = 0
                else:
                    matchFilterKernel[x][y] = -np.exp(-.5 * (x_ / sigma) ** 2) / (np.sqrt(2 * 180) * sigma)

        m = 0.0
        for i in range(matchFilterKernel.shape[0]):
            for j in range(matchFilterKernel.shape[1]):
                if matchFilterKernel[i][j] < 0:
                    m = m + 1
        mean = np.sum(matchFilterKernel) / m
        for i in range(matchFilterKernel.shape[0]):
            for j in range(matchFilterKernel.shape[1]):
                if matchFilterKernel[i][j] < 0:
                    matchFilterKernel[i][j] = (matchFilterKernel[i][j] - mean)
        filters.append((matchFilterKernel, theta))

    return filters


def get_max_filter_responses(img, filters):
    """Applies the provided filters to the supplied image. The maximum response
    obtained at each pixel as well as the orientation of the filter that
    obtained that reponse are recorded.

    Returns image containing the maximum response obtained by the filters at
    each pixel and image containing the orientation of the filter that
    obatained the maximum response at each pixel.

    Refer to section 4D of readme reference 2 for more information"""

    accum = np.zeros_like(img)
    direction_map = np.zeros_like(img)

    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8U, kern[0], borderType=cv2.BORDER_REPLICATE)
        for i in range(len(accum)):
            for j in range(len(accum[i])):
                if accum[i][j] < fimg[i][j]:
                    accum[i][j] = fimg[i][j]
                    direction_map[i][j] = kern[1]

    return accum, direction_map


class ProcessImage:
    def __init__(self, file_dir, output_dir):
        self.dir = file_dir
        # print(file_dir)
        self.image = cv2.imread(file_dir)
        self.output_dir = output_dir

    def mask_generation(self, image, scale_down=False):
        """
        Apply binary threshold of 35 to red band
        Apply open, then close, the erosion using 3x3 kernel
        """
        # image = cv2.resize(image, (150,150))
        if scale_down:
            image = cv2.pyrDown(image)
            image = cv2.pyrDown(image)
            image = cv2.pyrDown(image)
            # image = cv2.pyrDown(image)
        red_chanel = get_color_channel(img=image, channel='r')
        mask = binary_threshold(red_chanel)
        mask = morphological(mask)

        crop_info = crop_info_from_bounding_box_with_pad(mask, image)
        image = crop_image(image, crop_info)
        mask = crop_image(mask, crop_info)
        return image, mask
        # image[np.where((image == [0, 0, 0]).all(axis=2))] = [255, 255, 255] # this change background to white

    ########################################################################################################################

    def reduction_of_region_of_interest(self):
        erroded = errosion_circular(self.mask)
        image = cv2.bitwise_and(self.image1, self.image1, mask=erroded)
        image = resize(image, (1000, 1000))
        erroded = resize(erroded, (1000, 1000))
        # red = cv2.cvtColor(get_color_channel(image, 'r'), cv2.COLOR_GRAY2RGB)
        # green = cv2.cvtColor(get_color_channel(image, 'g'), cv2.COLOR_GRAY2RGB)
        # blue = cv2.cvtColor(get_color_channel(image, 'b'), cv2.COLOR_GRAY2RGB)
        # return np.hstack((image, red, green, blue)), 1
        return image, erroded

    def illumination_equalization(self):
        green_chanel = get_color_channel(self.image2, 'g')
        green_chanel = clahe_enhancement(green_chanel)
        green_chanel = equalize_illumination(green_chanel, 40, 45)
        green_chanel = equalize_illumination(green_chanel, 40, 45)

        blue_chanel = get_color_channel(self.image2, 'b')
        blue_chanel = clahe_enhancement(blue_chanel)
        blue_chanel = equalize_illumination(blue_chanel, 40, 45)
        blue_chanel = equalize_illumination(blue_chanel, 40, 45)

        blue_chanel_cropped = cv2.bitwise_and(blue_chanel, self.mask)
        green_chanel_cropped = cv2.bitwise_and(green_chanel, self.mask)
        # return np.hstack((blue_chanel_cropped, green_chanel_cropped))
        image = self.image2.copy()
        image[:, :, 1] = green_chanel_cropped
        image[:, :, 0] = blue_chanel_cropped
        return image

    def red_band_enhancement(self):
        image = self.image3.copy()
        blue_chanel = get_color_channel(self.image3, 'b')
        green_chanel = get_color_channel(self.image3, 'g')



        green_chanel = cv2.bitwise_and(green_chanel, self.mask)
        blue_chanel = cv2.bitwise_and(blue_chanel, self.mask)

        # blue_chanel = equalize_illumination(blue_chanel, 60, 45)
        # green_chanel = equalize_illumination(green_chanel, 60, 45)

        # Some funky formula
        inverse_green = 255 - green_chanel
        np.putmask(blue_chanel, inverse_green < blue_chanel, green_chanel)
        G = blue_chanel + green_chanel
        rt = np.mean(blue_chanel)

        b_enhanced = np.subtract(G.astype(np.int16), rt).clip(0, 255).astype(np.uint8)
        b_enhanced = cv2.bitwise_and(b_enhanced, self.mask)

        image[:, :, 0] = b_enhanced
        return b_enhanced

    def vessel_segmentation(self):

        self.blue_enhanced = clahe_enhancement(self.blue_enhanced)
        self.blue_enhanced = gamma_correction(self.blue_enhanced, gamma=5)
        self.blue_enhanced = cv2.bitwise_and(self.blue_enhanced, self.mask)

        filters = build_filters()
        gaussian_image, direction_map = get_max_filter_responses(self.blue_enhanced, filters)
        # filtering out the outer ring when thresholding
        gaussian_image_mask = cv2.bitwise_and(gaussian_image, gaussian_image, mask=self.mask)
        _, gaussian_image_mask = cv2.threshold(gaussian_image_mask, 5, 255, cv2.THRESH_OTSU)

        # reduce number of candidate pixel
        green_chanel = get_color_channel(self.image3, 'g')
        cnts = cv2.findContours(gaussian_image_mask, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            area = cv2.contourArea(c)
            if area < gaussian_image_mask.shape[0] * gaussian_image_mask.shape[1] / 450:
                cv2.drawContours(gaussian_image_mask, [c], -1, (0, 0, 0), cv2.FILLED)

        green_chanel = cv2.bitwise_and(green_chanel, gaussian_image_mask)


        # return np.hstack((self.blue_enhanced, gaussian_image_mask))

        return self.blue_enhanced

    def get_processed_image(self):
        self.image1, self.mask = self.mask_generation(self.image, scale_down=False)

        self.image2, self.mask = self.reduction_of_region_of_interest()
        # return self.image2
        self.image3 = self.illumination_equalization()
        self.blue_enhanced = self.red_band_enhancement()
        # return self.blue_enhanced
        self.image5 = self.vessel_segmentation()
        return self.image5
