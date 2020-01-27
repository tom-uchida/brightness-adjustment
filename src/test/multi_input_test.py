import cv2, matplotlib
import numpy as np
import glob
image_files = glob.glob("images/serial_number_images/*.bmp")


# ----------------------------
# ----- Read input image -----
# ----------------------------
def read_img(_img_name):
	# read input image
	img = cv2.imread(_img_name)

	# convert color (BGR → RGB)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	return img

# Read serial number images
for i in range(10):
    imgs = read_img("images/serial_number_images/image{0:03d}.bmp".format(i))

# for i in image_files:
#     imgs = cv2.imread(i)


print("ndim     : ", imgs.ndim)
print("shape    : ", imgs.shape)
print("dtype    : ", imgs.dtype)