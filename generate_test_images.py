import argparse
from itertools import product

import imageio
import numpy as np
from numpy import array, gradient
from skimage.draw import disk, rectangle
from skimage.filters import gaussian


def normalize_image(img):
    return (img - img.min()) / (img.max()-img.min())


def shade(i, f, relative=False, dimensions=2):
    scale = 1.0 / np.array(i.shape[:dimensions]) if relative else 1.0
    for p in product(*map(range, i.shape[:dimensions])):
        i[p] = f(array(p) * scale)


def shade_new(shape, f, dtype=np.float64, relative=False, dimensions=2):
    i = np.zeros(shape, dtype=dtype)
    shade(i, f, relative, dimensions)
    return i


def displacement_magnitude(displacement_field):
    return np.sqrt(displacement_field[..., 0] ** 2 + displacement_field[..., 1] ** 2)


def generate_checkerboard_from_mesh(y, x, interval):
    return 0.5 + (np.cos(x * np.pi / interval[0]) + np.cos(y * np.pi / interval[1])) / 4.0


def generate_height_field_ripples(size):
    def height_at_point(p, wave_center, wave_interval, wave_amplitude=0.01, wave_falloff=10):
        r = np.sqrt(sum((p - wave_center) ** 2))
        height = wave_amplitude * np.cos(r * np.pi / wave_interval) * (0.05 / ((r + 0.5) ** wave_falloff + 0.2))
        return height

    return shade_new(size, lambda p: height_at_point(p,
                                                     wave_center=array([0.5, 0.5]),
                                                     wave_interval=0.1,
                                                     wave_amplitude=0.1),
                     relative=True)


def generate_height_field_smiley(size):
    result = np.zeros(size, dtype=float)
    result[disk((size[0] / 2, size[1] / 2), size[0] / 3, shape=size)] = 0.04
    result[disk((size[0] / 2, size[1] / 2), size[0] / 4, shape=size)] = 0
    result[tuple(rectangle(start=(0,0), end=(size[0]//2, size[1]), shape=size))] = 0

    result[disk((size[0] / 5, size[1] / 3), size[0] / 10, shape=size)] = 0.07
    result[disk((size[0] / 5, 2 * size[1] / 3), size[0] / 10, shape=size)] = 0.07


    result = gaussian(result, 10)
    return result


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('pattern', choices=['ripples', 'smiley'], default='ripples')
    argparser.add_argument('--size_x', type=int, default=256)
    argparser.add_argument('--size_y', type=int, default=256)
    argparser.add_argument('--gain', type=float, default=10)
    argparser.add_argument('--frequency', type=float, default=30)

    args = argparser.parse_args()

    size = (args.size_y, args.size_y)

    height_field = {'ripples': generate_height_field_ripples,
                    'smiley': generate_height_field_smiley}[args.pattern](size)

    height_field_gradient_y, height_field_gradient_x  = gradient(height_field)

    pattern_interval = array([1.0/args.frequency, 1.0/args.frequency])

    y, x = np.meshgrid(*[np.linspace(0.0, 1, s) for s in size], indexing='ij')
    displaced_checkerboard = generate_checkerboard_from_mesh(y + args.gain*height_field_gradient_y,
                                                             x + args.gain*height_field_gradient_x,
                                                             pattern_interval)

    reference_checkerboard = generate_checkerboard_from_mesh(x, y, pattern_interval)

    imageio.imwrite('test image reference.tiff', (reference_checkerboard * 255).astype(np.uint8))
    imageio.imwrite('test image displaced.tiff', (displaced_checkerboard * 255).astype(np.uint8))
    imageio.imwrite('test image height field.tiff', (normalize_image(height_field) * 255).astype(np.uint8))
