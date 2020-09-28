import argparse
import cv2
import numpy


def init_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-first_image_path', default='output/grayscale_manual.png', type=str)
    parser.add_argument('-second_image_path', default='output/grayscale_cv2.png', type=str)
    parser.add_argument('--fast', action='store_true')
    return parser


def compare(first_image, second_image):
    result = 0
    for y in range(first_image.shape[0]):
        for x in range(first_image.shape[1]):
            result += (float(first_image[y, x]) - float(second_image[y, x])) ** 2
    result /= (first_image.shape[0] * first_image.shape[1])
    return result


def compare_fast(first_image, second_image):
    result = numpy.sum((first_image.astype(float) - second_image.astype(float)) ** 2)
    result = result / (first_image.shape[0] * first_image.shape[1])
    return result


def main(args):
    # Открытие изображений
    first_image = cv2.imread(args.first_image_path, cv2.IMREAD_GRAYSCALE)
    second_image = cv2.imread(args.second_image_path, cv2.IMREAD_GRAYSCALE)

    # Вычисление метрики сходства методом MSE
    if args.fast:
        mse = compare_fast(first_image, second_image)
    else:
        mse = compare(first_image, second_image)
    print('MSE =', mse)


arg_parser = init_arg_parser()
main(arg_parser.parse_args())
