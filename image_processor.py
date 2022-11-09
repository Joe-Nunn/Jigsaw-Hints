"""
This program processes an image of a jigsaw piece.
It removes the background of the image and ensures that the image has the right size.
"""
from rembg import remove
from PIL import Image

import cv2


class ImageProcessor:

    def resize_preserve_ratio(self, image_path, desired_size):
        img = cv2.imread(image_path)
        width = img.shape[1]
        height = img.shape[0]
        print('Before Resizing...')
        print('Image Width is', width)
        print('Image Height is', height)

        # Preserving the aspect ratio
        if width > desired_size and height > desired_size:
            scale = 1 - ((width - desired_size) / width)

            img_resized = cv2.resize(img, None, fx=scale, fy=scale)

            print('Resized!')
            print('Image Width is now', img_resized.shape[1])
            print('Image Height is now', img_resized.shape[0])

            cv2.imwrite(self.get_output_path(image_path, '-resized'), img_resized)

    def crop(self, image_path, desired_size):
        img = cv2.imread(image_path)
        y = 0
        x = 0
        height = desired_size
        width = desired_size
        print('Before Cropping...')
        print('Image Width is', img.shape[1])
        print('Image Height is', img.shape[0])
        img_cropped = img[y:y + height, x:x + width]
        print('Cropped!')
        print('Image Width is now', img_cropped.shape[1])
        print('Image Height is now', img_cropped.shape[0])
        cv2.imwrite(self.get_output_path(image_path, '-cropped'), img_cropped)

    def remove_background(self, image_path):
        input = Image.open(image_path)
        output = remove(input)
        output.save(self.get_output_path(image_path, '-bg-removed'))
        print('Background Removed! : ', output_path)

    def get_output_path(self, input_path, type):
        idx = input_path.index('.png')
        return input_path[:idx] + type + input_path[idx:]


"""
Main method.
"""


def main():
    """
    Main
    """

    desired_size = 256

    image_path = 'images/test.png'

    processor = ImageProcessor()

    processor.crop(image_path, desired_size)

    processor.resize_preserve_ratio(image_path, desired_size)

    print("Done!")


if __name__ == "__main__":
    main()
