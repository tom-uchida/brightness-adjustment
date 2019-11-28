import numpy as np
import cv2
import glob
import sys



# ----------------------------
# ----- Read Input Image -----
# ----------------------------
def read_img(_img_name):
    # read input image
    img_in_BGR = cv2.imread(_img_name)

    # convert color (BGR → RGB)
    img_in_RGB = cv2.cvtColor(img_in_BGR, cv2.COLOR_BGR2RGB)

    return img_in_RGB



# --------------------------------------------
# ----- Correct pixel value for each RGB -----
# --------------------------------------------
def correct_pixel_value(_rgb_img, _param):
    # Multiply
    red   = cv2.multiply(_rgb_img[:, :, 0], _param) # R
    green = cv2.multiply(_rgb_img[:, :, 1], _param) # G
    blue  = cv2.multiply(_rgb_img[:, :, 2], _param) # B

    # Apply correction
    corrected_img_RGB = np.empty((_rgb_img.shape[0], _rgb_img.shape[1], 3), dtype=np.uint8)
    corrected_img_RGB[:, :, 0] = red
    corrected_img_RGB[:, :, 1] = green
    corrected_img_RGB[:, :, 2] = blue

    return corrected_img_RGB



# ------------------------------
# ----- Write output Image -----
# ------------------------------
def write_img(_img_name, _i):
    # convert color (RGB → BGR)
    img_out_BGR = cv2.cvtColor(_img_name, cv2.COLOR_RGB2BGR)

    img_name = "images/serial_number_images/corrected_image{0:03d}.bmp"

    cv2.imwrite(img_name.format(_i), img_out_BGR)

    return



# -------------------------------------------
# ----- Processing on input image(LR=1) -----
# -------------------------------------------
args = sys.argv
img_in_RGB_LR1 = read_img(args[1])

# From the input image with LR = 1, 
#   calc the ratio that the pixel is 255 after correction
img_in_LR1_gray         = cv2.cvtColor(img_in_RGB_LR1, cv2.COLOR_RGB2GRAY)
img_in_LR1_gray_nonzero = img_in_LR1_gray[img_in_LR1_gray > 0]
N_all_nonzero_LR1       = np.sum(img_in_LR1_gray_nonzero > 0)
ratio_overexpose        = round( np.sum(img_in_LR1_gray == 255) / N_all_nonzero_LR1, 5)
ratio_overexpose_per    = ratio_overexpose * 100

if ratio_overexpose_per < 0.01: # < 0.01(%)
    ratio_overexpose = 0.01 * 0.01 # 1.0e-04
    print("\n** Note :")
    print("** Set ratio_overexpose = 0.0001 (0.01%)")
    print("**  because in the input image with LR = 1 (", args[2], "),")
    print("**  the ratio of pixels that are overexposed is too small (< 0.01%).")

print("\nratio_overexpose\n>", ratio_overexpose, " (", round(ratio_overexpose*100, 3), "(%) )")



# -----------------------------------------------
# ----- Correct pixel value 
#           for all images in the directory -----
# -----------------------------------------------
#image_files = glob.glob("images/serial_number_images/*.bmp")

img_count = 0
#for i in image_files:
for i in range(180):
    # Read input image
    #img_in_RGB = read_img(i)
    img_in_RGB = read_img("images/serial_number_images/image{0:03d}.bmp".format(i))

    # Set initial parameter
    p_init = 1.0
    p_interval = 0.01

    # Calc all number of pixels of the input image
    N_all = img_in_RGB.shape[0]*img_in_RGB.shape[1]

    # Then, calc number of pixels that pixel value is 0
    img_in_Gray     = cv2.cvtColor(img_in_RGB, cv2.COLOR_RGB2GRAY)
    N_all_nonzero   = np.sum(img_in_Gray > 0)

    # Calc the theoretical number of pixels that the pixel value is 255 after correction
    N_theor = int(N_all_nonzero * ratio_overexpose)

    # Determine parameter
    p = p_init
    count_overexpose_255 = 0
    while count_overexpose_255 < N_theor:
        tmp_corrected_img_RGB = correct_pixel_value(img_in_RGB, p)
        tmp_corrected_img_Gray = cv2.cvtColor(tmp_corrected_img_RGB, cv2.COLOR_RGB2GRAY)

        # Count number of max pixel value(==255)
        count_overexpose_255 = np.sum(tmp_corrected_img_Gray == 255)

        # Update parameter
        p += p_interval

    p_final = round(p, 2)

    # Make output image
    img_out_RGB = correct_pixel_value(img_in_RGB, p_final)

    # Write output image
    write_img(img_out_RGB, img_count)

    # Update image count
    img_count += 1

    if i == 179:
        print("\nNumber of input images\n>", i+1)
        print("\n")