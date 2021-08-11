# Fast Checkboard Demodulation #
Fast Checkerboard Demodulation (FCD) for Synthetic Schlieren imaging, in python

Completely based the work from Sander Wildeman: https://github.com/swildeman/fcd

## Usage ##
To apply the FCD to a single or series of images, run `fcd.py` as a program:

Run `python fcd.py --help` to see how to use it.
```
usage: fcd.py [-h] [--output-format {tiff,bmp,png,jpg,jpeg}] [--skip-existing]
              output_folder reference_image definition_image
              [definition_image ...]

positional arguments:
  output_folder
  reference_image
  definition_image      May contain wildcards

optional arguments:
  -h, --help            show this help message and exit
  --output-format {tiff,bmp,png,jpg,jpeg}
                        The output format (default: tiff)
  --skip-existing       Skip processing an image if the output file already
                        exists (default: False)
```

### Example ###
`python fcd.py output input/frame_000.tiff input/frame_*.tiff`

## Dependencies ##
Python >= 3.7  and the following PIP packages:
* numpy
* scipy
* imageio
* skimage
* more_itertools


## About FCD ##
Information on the FCD technique and the employed algorithms can be found in: Wildeman S., *Real-time quantitative Schlieren imaging by fast Fourier demodulation of a checkered backdrop*, Exp. Fluids (2018) 59: 97, https://doi.org/10.1007/s00348-018-2553-9, or https://arxiv.org/abs/1712.05679 (preprint)