import time
from dataclasses import dataclass
from typing import List

import imageio
import numpy as np
from numpy import array
from scipy.fft import fft2, ifft2, ifftshift
from skimage.draw import disk
from more_itertools import flatten

from fft_inverse_gradient import fftinvgrad
from find_peaks import find_peaks
from kspace import pixel2kspace

def normalize_image(img):
    return (img - img.min()) / (img.max()-img.min())

def peak_mask(shape, pos, r):
    result = np.zeros(shape, dtype=np.bool8)
    result[disk(pos, r, shape=shape)] = True
    return result


def ccsgn(i_ref_fft, mask):
    return np.conj(ifft2(i_ref_fft * mask))


@dataclass
class Carrier:
    pixel_loc: array
    k_loc: array
    krad: float
    mask: array
    ccsgn: array


def calculate_carriers(i_ref):
    peaks = find_peaks(i_ref)
    peak_radius = np.linalg.norm(peaks[0] - peaks[1]) / 2
    i_ref_fft = fft2(i_ref)

    carriers = [Carrier(peak, pixel2kspace(i_ref.shape, peak), peak_radius, mask, ccsgn(i_ref_fft, mask)) for mask, peak
                in
                [(ifftshift(peak_mask(i_ref.shape, peak, peak_radius)), peak) for peak in peaks]]
    return carriers


def fcd(i_def, carriers: List[Carrier]):
    i_def_fft = fft2(i_def)

    phis = [-np.angle(ifft2(i_def_fft * c.mask) * c.ccsgn) for c in carriers]

    det_a = carriers[0].k_loc[1] * carriers[1].k_loc[0] - carriers[0].k_loc[0] * carriers[1].k_loc[1]
    u = (carriers[1].k_loc[0] * phis[0] - carriers[0].k_loc[0] * phis[1]) / det_a
    v = (carriers[0].k_loc[1] * phis[1] - carriers[1].k_loc[1] * phis[0]) / det_a

    return fftinvgrad(-u, -v)

if __name__ == "__main__":
    import argparse
    import glob
    from pathlib import Path

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('output_folder', type=Path)
    argparser.add_argument('reference_image', type=str)
    argparser.add_argument('definition_image', nargs='+', help='May contain wildcards')
    argparser.add_argument('--output-format', default='tiff', choices=['tiff', 'bmp', 'png', 'jpg', 'jpeg'], help='The output format')
    argparser.add_argument('--skip-existing', action='store_true', help='Skip processing an image if the output file already exists')

    args = argparser.parse_args()

    args.output_folder.mkdir(exist_ok=True)

    i_ref = imageio.imread(args.reference_image) / 255.0

    print(f'processing reference image...', end='')
    carriers = calculate_carriers(i_ref)
    print('done')

    files = list(sorted(flatten((glob.glob(x) if '*' in x else [x]) for x in args.definition_image)))

    for file in files:
        output_file_path = args.output_folder.joinpath(f'{Path(file).stem}.{args.output_format}')

        if args.skip_existing and output_file_path.exists():
            continue

        print(f'processing {file} -> {output_file_path} ... ', end='')
        i_def = imageio.imread(file) / 255.0
        t0 = time.time()
        height_field = fcd(i_def, carriers)
        print(f'done in {time.time() - t0:.2}s\n')

        imageio.imwrite(output_file_path, (normalize_image(height_field) * 255.0).astype(np.uint8))
