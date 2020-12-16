import argparse
import cv2
import numpy
import random


def init_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', default='input/image_small.png', type=str)
    parser.add_argument('-method', default='split', type=str)
    parser.add_argument('-i', default=0, type=int)
    parser.add_argument('-j', default=0, type=int)
    return parser


def filter_threshold(image):
    result_image = numpy.zeros((image.shape[0], image.shape[1]), numpy.ubyte)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y, x] > 128:
                result_image[y, x] = 255
    return result_image


def uniform(image, segment):
    color = image[segment[0], segment[1]]
    for y in range(segment[0], segment[2]):
        for x in range(segment[1], segment[3]):
            if image[y, x] != color:
                return False
    return True


def split(image):
    stack = [[0, 0, image.shape[0] , image.shape[1]]]
    result = []
    while True:
        can_exit = True
        for segment in stack:
            if not uniform(image, segment):
                y_b = segment[0]
                y_m = (segment[0] + segment[2]) // 2
                y_t = segment[2]
                x_b = segment[1]
                x_m = (segment[1] + segment[3]) // 2
                x_t = segment[3]

                stack.remove(segment)

                if y_b != y_m and x_b != x_m:
                    stack.append([y_b, x_b, y_m, x_m])
                if y_m != y_t and x_b != x_m:
                    stack.append([y_m, x_b, y_t, x_m])
                if y_b != y_m and x_b != x_t:
                    stack.append([y_b, x_m, y_m, x_t])
                if y_m != y_t and x_m != x_t:
                    stack.append([y_m, x_m, y_t, x_t])
                can_exit = False
            else:
                result.append(segment)
                stack.remove(segment)
                can_exit = False
                break
        if can_exit:
            break
    return result


def clamp(value, min_val, max_val):
    return max(min(value, max_val), min_val)


def p_uniform(image, pixels):
    color = image[pixels[0][0], pixels[0][1]]
    for pixel in pixels:
        if image[pixel[0], pixel[1]] != color:
            return False
    return True


def merge(image, in_segments=None):
    mask = numpy.zeros((image.shape[0], image.shape[1]), numpy.ulonglong)
    segments = {}
    borders = {}
    if in_segments is None:
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                mask[y, x] = y * mask.shape[1] + x
                segments[mask[y, x]] = [[y, x]]
                borders[mask[y, x]] = [[y, x]]
    else:
        segment_id = 0
        for segment in in_segments:
            segments[segment_id] = []
            borders[segment_id] = []
            for y in range(segment[0], segment[2]):
                for x in range(segment[1], segment[3]):
                    mask[y, x] = segment_id
                    segments[segment_id].append([y, x])
                    borders[segment_id].append([y, x])
            segment_id += 1

    while True:
        can_exit = True
        for segment in segments.items():
            # Поиск соседей
            segment_id = segment[0]
            neighbors = []
            for pixel in borders[segment[0]]:
                is_border = False
                for y in range(clamp(pixel[0] - 1, 0, image.shape[0]), clamp(pixel[0] + 2, 0, image.shape[0])):
                    for x in range(clamp(pixel[1] - 1, 0, image.shape[1]), clamp(pixel[1] + 2, 0, image.shape[1])):
                        if mask[y, x] != segment_id:
                            is_border = True
                            if not mask[y, x] in neighbors:
                                neighbors.append(mask[y, x])
                if not is_border:
                    borders[segment[0]].remove(pixel)
            # Проверка соседей
            for nei_id in neighbors:
                sum_list = []
                sum_list.extend(segment[1])
                sum_list.extend(segments[nei_id])
                if p_uniform(image, sum_list):
                    for pixel in segment[1]:
                        mask[pixel[0], pixel[1]] = nei_id
                    can_exit = False
                    segments[nei_id].extend(segment[1])
                    borders[nei_id].extend(borders[segment[0]])
                    borders.pop(segment[0])
                    segments.pop(segment[0])
                    break
            if not can_exit:
                break
        if can_exit:
            break
    return segments


def paint_segment(segments, shape):
    result_image = numpy.zeros((shape[0], shape[1], 3), numpy.ubyte)
    for segment in segments:
        color = [random.randint(64, 256), random.randint(48, 208), random.randint(64, 256)]
        for y in range(segment[0], segment[2]):
            for x in range(segment[1], segment[3]):
                result_image[y, x] = color
    return result_image


def p_paint_segment(segments, shape):
    result_image = numpy.zeros((shape[0], shape[1], 3), numpy.ubyte)
    for segment in segments.values():
        color = [random.randint(0, 256), random.randint(48, 208), random.randint(0, 256)]
        for pixel in segment:
            result_image[pixel[0], pixel[1]] = color
    return result_image


def moment(segments, image, coefs):
    result = []
    for segment in segments:
        seg_res = 0
        for y in range(segment[0], segment[2]):
            for x in range(segment[1], segment[3]):
                seg_res += image[y, x] * (y ** coefs[0]) * (x ** coefs[1])
        result.append(seg_res)
    return result


def p_moment(segments, image, coefs):
    result = []
    for segment in segments.items():
        seg_res = 0
        for pixel in segment[1]:
            seg_res += image[pixel[0], pixel[1]] * (pixel[0] ** coefs[0]) * (pixel[1] ** coefs[1])
        result.append(seg_res)
    return result


def main(args):
    origin_image = cv2.imread(args.path, cv2.IMREAD_GRAYSCALE)
    image = filter_threshold(origin_image)
    file = open('./output/moments.txt', 'w')
    moments = []

    if args.method == 'split':
        segments = split(image)
        segmented_image = paint_segment(segments, image.shape)
        moments = moment(segments, image, (args.i, args.j))
        cv2.imwrite('./output/split.png', segmented_image)
    elif args.method == 'merge':
        segments = merge(image)
        segmented_image = p_paint_segment(segments, image.shape)
        moments = p_moment(segments, image, (args.i, args.j))
        cv2.imwrite('./output/merge.png', segmented_image)
    elif args.method == 'combined':
        segments = split(image)
        segments = merge(image, segments)
        segmented_image = p_paint_segment(segments, image.shape)
        moments = p_moment(segments, image, (args.i, args.j))
        cv2.imwrite('./output/combined.png', segmented_image)

    for i in range(len(moments)):
        file.write('Moment of segment ' + str(i) + ' = ' + str(moments[i]) + '\n')


if __name__ == '__main__':
    arg_parser = init_arg_parser()
    main(arg_parser.parse_args())
