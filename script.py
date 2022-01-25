import cv2
import numpy as np
import click
import os

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
def long_exposure(
    input_path,
    output_path,
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
            out_image += i

    out_image /= len(images) / 15
    out_image = np.clip(out_image, 0, 255)
    out_image = out_image.astype(np.uint8)
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, out_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

if __name__ == '__main__':
    long_exposure()