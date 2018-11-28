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

# Initialize section ratio
specified_section_ratio = 0.01 # 1%

# Calc max pixel value of the input image(LR=1)
img_in_Gray_LR1     = cv2.cvtColor(img_in_RGB_LR1, cv2.COLOR_RGB2GRAY)
N_all_nonzero_LR1   = np.sum(img_in_Gray_LR1 > 0)
max_pixel_value_LR1 = np.max(img_in_Gray_LR1)
print("\nmax_pixel_value (LR=1)\n>", max_pixel_value_LR1, "(pixel value)")

# Search for pixel value that determines the specified section 
target_pixel_value  = max_pixel_value_LR1
tmp_ratio_LR1       = 0.0
while tmp_ratio_LR1 < specified_section_ratio:
    tmp_sum_pixel_number = np.sum( target_pixel_value <= img_in_Gray_LR1 )

    # Temporarily, calc specified section ratio
    tmp_ratio_LR1 = tmp_sum_pixel_number / N_all_nonzero_LR1

    # Next pixel value
    target_pixel_value -= 1

print("\n\n** Specified section was confirmed.")
standard_pixel_value = target_pixel_value
print("standard_pixel_value (LR=1)\n>", standard_pixel_value, "(pixel value)")

specified_section_ratio_LR1_final = tmp_ratio_LR1
print("\nspecified_section_ratio_LR1_final (LR=1)\n>", round(specified_section_ratio_LR1_final*100, 1), "(%) ( >=", standard_pixel_value, ")")



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

    # Determine parameter
    p = p_init
    tmp_ratio = 0.0
    while tmp_ratio < specified_section_ratio:
        tmp_corrected_img_RGB   = correct_pixel_value(img_in_RGB, p)
        tmp_corrected_img_Gray  = cv2.cvtColor(tmp_corrected_img_RGB, cv2.COLOR_RGB2GRAY)

        # Temporarily, calc specified section ratio (>= standard_pixel_value)
        tmp_sum_pixel_number = np.sum( standard_pixel_value <= tmp_corrected_img_Gray )
        # tmp_sum_pixel_number = np.sum( (standard_pixel_value <= tmp_corrected_img_Gray) & (tmp_corrected_img_Gray < 255) )
        tmp_ratio = tmp_sum_pixel_number / N_all_nonzero

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