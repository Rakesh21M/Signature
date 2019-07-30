# Code to extract signature from the scanned document.

import argparse
import numpy as np
from skimage import measure
from skimage import morphology
from skimage.measure import regionprops
import os
import cv2

current_path = os.getcwd()
test_path = "/TestOutput/"


def main(image):
    # Reading the image
    img = cv2.imread(image, 0)
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]

    # Converting to bool array and labeling with background white
    bool_array = img > img.mean()
    bool_label = measure.label(bool_array, background=1)

    # Calculating the largest connected components threshold
    # value to reject connected pixel smaller than threshold
    total_area, largest_comp, count = signature_ex(bool_label)
    average = total_area/count
    # print(average)
    connected_thresh = ((average/84.0)*250.0)+100

    im = morphology.remove_small_objects(bool_label, connected_thresh)

    # Creating a folder to store our output
    """
    im.jpg is a negative image of signature
    fimg.jpg is a image with the signature
    """
    if not os.path.exists(current_path+test_path):
        os.makedirs(current_path+test_path)
    cv2.imwrite(current_path+test_path+'im.jpg', im)

    fimg = cv2.imread(current_path+test_path+'im.jpg', 0)
    fimg = cv2.threshold(fimg, 0, 255,
                         cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    cv2.imwrite(current_path+test_path+'fimg.jpg', fimg)
    image = cv2.imread(current_path+test_path+'fimg.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    extra_line_rm(image)


def signature_ex(bool_label):
    """

    :param bool_label: labeled input
    :return: largest component and the parameter
    to calculate the longest connected pixel threshold
    """
    largest_comp = 0
    count = 1
    total_area = 0.0

    for region in regionprops(np.squeeze(bool_label)):
        if region.area > 10:
            total_area = total_area + region.area
            count = count + 1

        if region.area >= 250 and (region.area > largest_comp):
            largest_comp = region.area

    return total_area, largest_comp, count


def extra_line_rm(image):
    """

    :param image: image with only signature, lines, symbol
    :return: None
    """
    img = cv2.bitwise_not(image)
    th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 15, -2)

    horizontal = th2
    vertical = th2
    rows, cols = horizontal.shape
    horizontal_size = int(cols / 30)
    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT,
                                                     (horizontal_size, 1))
    horizontal = cv2.erode(horizontal,
                           horizontal_structure, (-1, -1))

    horizontal = cv2.dilate(horizontal,
                            horizontal_structure, (-1, -1))

    horizontal = cv2.bitwise_not(horizontal)

    vertical_size = int(rows / 30)
    vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT,
                                                   (1, vertical_size))
    vertical = cv2.erode(vertical, vertical_structure, (-1, -1))
    vertical = cv2.dilate(vertical, vertical_structure, (-1, -1))
    vertical = cv2.bitwise_not(vertical)

    line_remove(vertical, horizontal, img)


def line_remove(vertical, horizontal, img):
    """

    :param vertical: The image which only has vertical lines
    :param horizontal: The image which only has horizontal lines
    :param img: The image which contain lines, signature
    :return: None
    """
    image = cv2.bitwise_not(img)
    final_image = cv2.addWeighted(horizontal, 0.5, vertical, 0.5, 0)
    _, th_1 = cv2.threshold(final_image, 220, 255, cv2.THRESH_BINARY)
    _, th_2 = cv2.threshold(image, 220, 255, cv2.THRESH_BINARY)
    th_1 = abs(255-th_1)
    th_2 = abs(255-th_2)
    final_image1 = th_2-th_1
    kernel = np.ones((2, 2), dtype="uint8")
    erode1 = cv2.erode(final_image1, kernel)
    final_image2 = cv2.bitwise_not(erode1)

    # This final_image.jpg is the final output
    cv2.imwrite(current_path+test_path+'final_image.jpg', final_image2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Taking the image through command line
    parser.add_argument('-i', '--image', help="Path of image is required",
                        required=True)
    arg = parser.parse_args()

    main(arg.image)
