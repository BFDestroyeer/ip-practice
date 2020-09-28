import argparse
import cv2
import numpy


def init_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', default='input/cat.png', type=str)
    return parser


def rgb_to_yuv(image):
    result_image = numpy.zeros((image.shape[0], image.shape[1], 3), numpy.ubyte)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            result_image[y, x, 0] = 0.299 * image[y, x, 0] + 0.587 * image[y, x, 1] + 0.114 * image[y, x, 2]
            result_image[y, x, 1] = - 0.147 * image[y, x, 0] - 0.289 * image[y, x, 1] + 0.436 * image[y, x, 2] + 128
            result_image[y, x, 2] = 0.615 * image[y, x, 0] - 0.515 * image[y, x, 1] - 0.100 * image[y, x, 2] + 128
    return result_image


def main(args):
    image = cv2.imread(args.path)

    # Конвертация в YUV средсвами OpenCV
    cv2_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    cv2.imwrite("output/yuv_cv2.png", cv2_image)

    # Конвертация в YUV
    manual_image = rgb_to_yuv(image)
    cv2.imwrite("output/yuv_manual.png", manual_image, )


arg_parser = init_arg_parser()
main(arg_parser.parse_args())