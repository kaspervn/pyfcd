from itertools import product

import imageio
import numpy as np
from numpy import array

def normalize_image(img):
    return (img - img.min()) / (img.max()-img.min())

def shade(i, f, relative=False, dimensions=2):
    scale = 1.0 / np.array(i.shape[:dimensions]) if relative else 1.0
    for p in product(*list(map(range, i.shape[:dimensions]))):
        i[p] = f(array(p) * scale)

def shade_new(shape, f, dtype=np.float64, relative=False, dimensions=2):
    i = np.zeros(shape, dtype=dtype)
    shade(i, f, relative, dimensions)
    return i

def rotate_mesh(angle, x, y):
    return np.cos(angle) * x + np.sin(angle) * y , np.sin(angle) * x + np.cos(angle) * y

def translate_mesh(p, x, y):
    return x + p[0], y + p[1]

def wave_displacement(p, center, interval, amplitude=0.01, falloff=10):
    r = np.sqrt(sum((p - center) ** 2))
    displacement = amplitude * np.cos(r * np.pi / interval) * (0.05 / ((r + 0.5) ** falloff + 0.2))
    return displacement


def displacement_magnitude(displacement_field):
    return np.sqrt(displacement_field[..., 0] ** 2 + displacement_field[..., 1] ** 2)


def generate_checkerboard_from_mesh(x, y, interval):
    return 0.5 + (np.cos(x * np.pi / interval[0]) + np.cos(y * np.pi / interval[1])) / 4.0


size = (256, 256)

y, x = np.meshgrid(*[np.linspace(0.0, 1, s) for s in size], indexing='ij')

wave_displacement_field = shade_new((size[1], size[0], 2), lambda p: wave_displacement(p,
                                                                                       center=array([0.5, 0.5]),
                                                                                       interval=array([0.1, 0.1]),
                                                                                       amplitude=0.1),
                                    relative=True)

displaced_checkerboard = generate_checkerboard_from_mesh(x + wave_displacement_field[..., 0],
                                                         y + wave_displacement_field[..., 1],
                                                         array([0.03, 0.03]))

reference_checkerboard = generate_checkerboard_from_mesh(x, y, array([0.03, 0.03]))


imageio.imwrite('test_wave_reference.tiff', (reference_checkerboard * 255).astype(np.uint8))
imageio.imwrite('test_wave_displaced.tiff', (displaced_checkerboard * 255).astype(np.uint8))

displacement_mag = displacement_magnitude(wave_displacement_field)
imageio.imwrite('test_wave_displacement_magn.tiff', (normalize_image(displacement_magnitude(wave_displacement_field)) * 255).astype(np.uint8))