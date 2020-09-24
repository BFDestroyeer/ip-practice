import argparse
import cv2
import numpy
import os


def init_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-path',  default='input/cat.png', type=str)
    return parser


def main(image_path):
    image = cv2.imread(image_path)

    # Конвертация средствами OpenCV
    cv2_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('output/grayscale_cv2.png', cv2_image)

    # Ручная конвертация методом 4 (Photoshop, GIMP)
    manual_image = numpy.dot(image, [0.3, 0.59, 0.11])
    manual_image = manual_image.astype(numpy.ubyte)
    cv2.imwrite('output/grayscale_manual.png', manual_image)

    # Вывод метрики сравнения
    os.system('python comparison.py '
              '-first_image_path "output/grayscale_cv2.png" '
              '-second_image_path "output/grayscale_manual.png"')


arg_parser = init_arg_parser()
args = arg_parser.parse_args()
main(args.path)
