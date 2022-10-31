"""
This program processes an image of a jigsaw piece.
It removes the background of the image and ensures that the image has the right size.
"""
from rembg import remove
from PIL import Image

import cv2

desired_size = 256

image_path = 'images/test.png'

def main():
    """
    Main
    """

    resize(image_path, desired_size)

    remove_background(image_path)

    print("Done!")


def resize(image_path, desired_size):
    img = cv2.imread(image_path)
    width = img.shape[1]
    height = img.shape[0]
    print('Before Resizing...')
    print('Image Width is', width)
    print('Image Height is', height)

    # Preserving the aspect ratio
    # shorter_side = width
    # if width > desired_size and height > desired_size:
    #     if (height < width):
    #         shorter_side = height
    # img_75 = cv2.resize(img, None, fx=0.75, fy=0.75)

    img_resized = cv2.resize(img, (desired_size, desired_size))

    print('Resized!')
    print('Image Width is now', img_resized.shape[1])
    print('Image Height is now', img_resized.shape[0])

    cv2.imwrite(image_path, img_resized)


def remove_background(image_path):
    input_path = image_path
    idx = input_path.index('.png')
    output_path = input_path[:idx] + '-bg-removed' + input_path[idx:]
    input = Image.open(image_path)
    output = remove(input)
    output.save(output_path)
    print('Background Removed! : ', output_path)


if __name__ == "__main__":
    main()
