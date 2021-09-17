import imageio
import numpy as np
from numpy import array
from scipy.fft import fft2, fftshift, fftfreq
from skimage.measure import regionprops, label

from kspace import pixel2kspace


def peaks(img: array, thresshold, no_peaks, subpixel=False):
    if subpixel:
        raise NotImplementedError()

    blob_img = img > thresshold

    # make the borders false
    blob_img[0] *= False
    blob_img[-1] *= False
    blob_img[..., 0] *= False
    blob_img[..., -1] *= False

    blob_data = regionprops(label(blob_img.astype(np.uint8)))

    def blob_max_pixel_intensity(blob):
        pixels_with_coords = [(img[tuple(c)], c) for c in blob.coords]
        return max(pixels_with_coords, key=lambda x: x[0])

    blobs_with_max_intensity_and_coord = [blob_max_pixel_intensity(blob) for blob in blob_data]
    sorted_blobs = sorted(blobs_with_max_intensity_and_coord, key=lambda x: x[0])
    return [peak[1] for peak in sorted_blobs[:no_peaks]] # return the coordinates


def find_peaks(img):
    i_fft = fftshift(np.abs(fft2(img - np.mean(img))))

    #TODO: optimize by doing calculating the radius in pixels, and use the disc function in skimage.draw, like in fcd.py:mask()
    def highpass_mask():
        k_space_rows = fftshift(fftfreq(i_fft.shape[0], 1 / (2.0 * np.pi)))
        k_space_cols = fftshift(fftfreq(i_fft.shape[1], 1 / (2.0 * np.pi)))
        kmin = 4 * np.pi / min(img.shape)
        wavenumbers_mesh = np.meshgrid(k_space_rows, k_space_cols, indexing='ij')
        return (wavenumbers_mesh[0]**2 + wavenumbers_mesh[1]**2) > kmin**2

    i_fft *= highpass_mask()
    threshold = 0.5 * np.max(i_fft)

    peak_locations = peaks(i_fft, threshold, 4)
    rightmost_peak = min(peak_locations, key=lambda p: abs(np.arctan2(*pixel2kspace(i_fft.shape, p))))
    perpendicular_peak = min(peak_locations, key=lambda p: abs(np.dot(pixel2kspace(i_fft.shape, p), pixel2kspace(i_fft.shape, rightmost_peak))))

    return rightmost_peak, perpendicular_peak
