import argparse
import cv2
import numpy
import random
import time

import comparison


def init_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', default='input/cat.png', type=str)
    parser.add_argument('-intensity', default=1.0, type=float)
    return parser


def salt_and_pepper_noise(image, intensity):
    result_image = image.copy()
    noise_map = numpy.random.uniform(0, 100, (image.shape[0], image.shape[1]))
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if noise_map[y, x] < intensity:
                result_image[y, x] = (0, 0, 0)
            elif noise_map[y, x] > (99 - intensity):
                result_image[y, x] = (255, 255, 255)
    return result_image


def averaging_filter(image):
    result_image = numpy.zeros((image.shape[0], image.shape[1], 3), numpy.ubyte)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            blue = 0
            green = 0
            red = 0
            for local_y in range(max(0, y - 1), min(image.shape[0], y + 2)):
                for local_x in range(max(0, x - 1), min(image.shape[1], x + 2)):
                    blue += image[local_y, local_x, 0] / 9
                    green += image[local_y, local_x, 1] / 9
                    red += image[local_y, local_x, 2] / 9
            result_image[y, x, 0] = blue
            result_image[y, x, 1] = green
            result_image[y, x, 2] = red
    return result_image


def median_filter(image):
    result_image = numpy.zeros((image.shape[0], image.shape[1], 3), numpy.ubyte)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            blue = []
            green = []
            red = []
            for local_y in range(max(0, y - 1), min(image.shape[0], y + 2)):
                for local_x in range(max(0, x - 1), min(image.shape[1], x + 2)):
                    blue.append(image[local_y, local_x, 0])
                    green.append(image[local_y, local_x, 1])
                    red.append(image[local_y, local_x, 2])
            blue.sort()
            green.sort()
            red.sort()
            result_image[y, x] = (blue[len(blue) // 2], green[len(green) // 2], red[len(red) // 2])
    return result_image


def main(args):
    image = cv2.imread(args.path)

    noised_image = salt_and_pepper_noise(image, args.intensity)
    cv2.imwrite('output/noised.png', noised_image)

    begin_median = time.time()
    manual_median = median_filter(noised_image)
    cv2.imwrite('output/manual_median.png', manual_median)
    end_median = time.time()

    begin_averaging = time.time()
    manual_averaging = averaging_filter(noised_image)
    cv2.imwrite('output/manual_averaging.png', manual_averaging)
    end_averaging = time.time()

    begin_cv2 = time.time()
    cv2_image = cv2.medianBlur(noised_image, 3)
    cv2.imwrite('output/cv2_median.png', cv2_image)
    end_cv2 = time.time()

    print('Median filter time: ', end_median - begin_median)
    print('Averaging filter time: ', end_averaging - begin_averaging)
    print('OpenCV Median filter time', end_cv2 - begin_cv2)

    print('Median filter MSE: ', comparison.compare_color_fast(image, manual_median))
    print('Averaging filter MSE: ', comparison.compare_color_fast(image, manual_averaging))
    print('OpenCV Median filter MSE: ', comparison.compare_color_fast(image, cv2_image))


if __name__ == '__main__':
    arg_parser = init_arg_parser()
    main(arg_parser.parse_args())
