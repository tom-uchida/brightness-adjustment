import cv2
import numpy as np
import sys

# Check arguments
args = sys.argv
if len(args) != 2:
    #raise Exception
    sys.exit()

def read_image(_img_name):
    img_BGR = cv2.imread(_img_name)
    img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

    return img_RGB
# End of read_image()



if __name__ == "__main__":
    # Read an input image
    img_in_RGB = read_image(args[1])

    # Convert RGB to Grayscale
    img_in_Gray = cv2.cvtColor(img_in_RGB, cv2.COLOR_RGB2GRAY)

    # Otsu method
    threshold, img_in_otsu = cv2.threshold(img_in_Gray, 0, 255, cv2.THRESH_OTSU)

    print("Threshold pixel value = ", threshold)

    # Write an output image
    cv2.imwrite("IMAGE_DATA/otsu_method_test.png", img_in_otsu)