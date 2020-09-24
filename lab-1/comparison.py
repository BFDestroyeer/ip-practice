import argparse
import cv2
import numpy


def init_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-first_image_path',  default='output/grayscale_manual.png', type=str)
    parser.add_argument('-second_image_path', default='output/grayscale_cv2.png',    type=str)
    return parser


def main(first_image_path, second_image_path):
    # Открытие изображений
    first_image = cv2.imread(first_image_path, cv2.IMREAD_GRAYSCALE)
    second_image = cv2.imread(second_image_path, cv2.IMREAD_GRAYSCALE)

    # Вычисление метрики сходства методом MSE
    result = numpy.sum((first_image.astype(float) - second_image.astype(float)) ** 2)
    result = result / (first_image.shape[0] * first_image.shape[1])
    print('MSE =', result)


arg_parser = init_arg_parser()
args = arg_parser.parse_args()
main(args.first_image_path, args.second_image_path)
