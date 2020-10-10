import argparse
import cv2
import numpy

import comparison


def init_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', default='input/cat.png', type=str)
    return parser


def bgr_to_yuv(image):
    result_image = numpy.zeros((image.shape[0], image.shape[1], 3), numpy.ubyte)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            result_image[y, x, 0] = 0.299 * image[y, x, 2] + 0.587 * image[y, x, 1] + 0.114 * image[y, x, 0]
            result_image[y, x, 1] = - 0.147 * image[y, x, 2] - 0.289 * image[y, x, 1] + 0.436 * image[y, x, 0] + 128
            result_image[y, x, 2] = 0.615 * image[y, x, 2] - 0.515 * image[y, x, 1] - 0.100 * image[y, x, 0] + 128
    return result_image


def yuv_to_bgr(image):
    result_image = numpy.zeros((image.shape[0], image.shape[1], 3), numpy.ubyte)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            result_image[y, x, 0] = image[y, x, 0] + 2.032 * (image[y, x, 1] - 128)
            result_image[y, x, 1] = image[y, x, 0] - 0.395 * (image[y, x, 1] - 128) - 0.581 * (image[y, x, 2] - 128)
            result_image[y, x, 2] = image[y, x, 0] + 1.140 * (image[y, x, 2] - 128)
    return result_image


def increase_brightness_bgr(image, coefficient):
    result_image = numpy.zeros((image.shape[0], image.shape[1], 3), numpy.ubyte)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            result_image[y, x, 0] = min(image[y, x, 0] * coefficient, 255)
            result_image[y, x, 1] = min(image[y, x, 1] * coefficient, 255)
            result_image[y, x, 2] = min(image[y, x, 2] * coefficient, 255)
    return result_image


def increase_brightness_yuv(image, coefficient):
    result_image = numpy.zeros((image.shape[0], image.shape[1], 3), numpy.ubyte)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            result_image[y, x, 0] = min(image[y, x, 0] * coefficient, 255)
    return result_image


def main(args):
    image = cv2.imread(args.path)

    # Конвертация в YUV средсвами OpenCV
    cv2_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    cv2.imwrite("output/yuv_cv2.png", cv2_image)

    # Конвертация в YUV в форме YCbCr
    manual_image = bgr_to_yuv(image)
    cv2.imwrite("output/yuv_manual.png", manual_image, )

    # Сравнение результатов конвертации
    #mse = comparison.compare_color(cv2_image, manual_image)
    #print('YUV conversion MSE =', mse)

    # Повышение яроксти
    brighter_image_bgr = increase_brightness_bgr(image, 1.2)

    brighter_image_yuv = increase_brightness_yuv(manual_image, 1.2)
    brighter_image_yuv = yuv_to_bgr(brighter_image_yuv)

    cv2.imwrite("output/brighter_bgr.png", brighter_image_bgr)
    cv2.imwrite("output/brighter_yuv.png", brighter_image_yuv)


arg_parser = init_arg_parser()
main(arg_parser.parse_args())
