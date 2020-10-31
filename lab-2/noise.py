import argparse
import cv2
import numpy
import time


def init_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', default='input/cat.png', type=str)
    return parser


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

    begin_median = time.time()
    manual_image = median_filter(image)
    cv2.imwrite('output/manual_median.png', manual_image)
    end_median = time.time()

    begin_averaging = time.time()
    manual_image = averaging_filter(image)
    cv2.imwrite('output/manual_averaging.png')
    end_averaging = time.time()

    begin_cv2 = time.time()
    cv2_image = cv2.medianBlur(image, 3)
    cv2.imwrite('output/cv2_median.png', cv2_image)
    end_cv2 = time.time()


if __name__ == '__main__':
    arg_parser = init_arg_parser()
    main(arg_parser.parse_args())
