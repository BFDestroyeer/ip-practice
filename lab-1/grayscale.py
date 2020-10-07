import argparse
import cv2
import numpy

import comparison


def init_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-path',  default='input/cat.png', type=str)
    parser.add_argument('--fast', action='store_true')
    return parser


def manual_conversion(image):
    result_image = numpy.zeros((image.shape[0], image.shape[1]), numpy.ubyte)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            result_image[y, x] = image[y, x, 0] * 0.3 + image[y, x, 1] * 0.59 + image[y, x, 2] * 0.11
    return result_image


def manual_conversion_fast(image):
    result_image = numpy.dot(image, [0.3, 0.59, 0.11])
    result_image = result_image.astype(numpy.ubyte)
    return result_image


def main(args):
    image = cv2.imread(args.path)

    # Конвертация средствами OpenCV
    cv2_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('output/grayscale_cv2.png', cv2_image)

    # Ручная конвертация методом 4 (Photoshop, GIMP)
    if args.fast:
        manual_image = manual_conversion_fast(image)
    else:
        manual_image = manual_conversion(image)
    cv2.imwrite('output/grayscale_manual.png', manual_image)

    comparison.compare(manual_image, cv2_image)


arg_parser = init_arg_parser()
main(arg_parser.parse_args())
