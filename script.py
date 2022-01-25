import cv2
import numpy as np
import click
import os

def easeInExpo(x: np.ndarray):
    ret = np.full_like(x, 2)
    ret = np.power(ret, 10 * x - 10)
    mask = x == 0
    ret[mask] = 0
    return ret

@click.command()
@click.option(
    '-i',
    '--input_path',
    type=click.Path(exists=True),
    required=True,
    help='Path to folder of images.',
)
@click.option(
    '-o',
    '--output_path',
    type=click.Path(exists=False),
    required=True,
    help='Path of resulting image. Must end in .png. Necessary folder created automatically.',
)
@click.option(
    '-e',
    '--exposure',
    type=float,
    help='Increase or decrease exposure. 1.0 is no change.',
    default=1.0,
    show_default=True
)
def long_exposure(
    input_path,
    output_path,
    exposure,
):

    if output_path.split('.')[-1] != 'png':
        raise Exception(f'output_path must end in .png! path provided: {output_path}')

    images = []

    for root, _, files in os.walk(input_path):
        for f in files:
            path = os.path.join(root, f)
            print(f'loading {path}')

            images.append(cv2.imread(path))

    assert(len(images) > 0)
    shape = images[0].shape

    out_image = np.zeros((shape[0], shape[1], 3), np.float32)

    for i in images:
        if i.shape != shape:
            print(f'WARNING: {i.shape} does not equal first image shape. Skipping')
        else:
            img = i.astype(np.float32) / 255.0
            if exposure > 1.0:
                img *= 1 + (exposure / 10)
            img = easeInExpo(img)
            out_image += img

    out_image /= len(images)
    out_image *= exposure
    out_image *= 255
    out_image = np.clip(out_image, 0, 255)
    out_image = out_image.astype(np.uint8)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, out_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

if __name__ == '__main__':
    long_exposure()