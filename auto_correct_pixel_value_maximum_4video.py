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
    # Apply correction
    corrected_img_RGB = np.empty((_rgb_img.shape[0], _rgb_img.shape[1], 3), dtype=np.uint8)
    corrected_img_RGB[:, :, 0] = cv2.multiply(_rgb_img[:, :, 0], _param) # R
    corrected_img_RGB[:, :, 1] = cv2.multiply(_rgb_img[:, :, 1], _param) # G
    corrected_img_RGB[:, :, 2] = cv2.multiply(_rgb_img[:, :, 2], _param) # B

    return corrected_img_RGB



# ------------------------------
# ----- Write output Image -----
# ------------------------------
def write_img(_img_name, _i):
    # convert color (RGB → BGR)
    img_out_BGR = cv2.cvtColor(_img_name, cv2.COLOR_RGB2BGR)

    img_name = "images/serial_number_images/adjusted_image{0:03d}.bmp"

    cv2.imwrite(img_name.format(_i), img_out_BGR)

    return



# -------------------------------------------
# ----- Processing on input image(LR=1) -----
# -------------------------------------------
reference_section = 0.01 # 1%

args = sys.argv
img_in_RGB_LR1 = read_img(args[1])

# Calc max pixel value of the input image(LR=1)
img_in_Gray_LR1     = cv2.cvtColor(img_in_RGB_LR1, cv2.COLOR_RGB2GRAY)
N_all_nonzero_LR1   = np.sum(img_in_Gray_LR1 > 0)
max_pixel_value_LR1 = np.max(img_in_Gray_LR1)
print("\n-----", args[1], "-----")
print("Max pixel value\n>", max_pixel_value_LR1, "(pixel value)")

# Calc the ratio of the maximum pixel value
ratio_max_pixel_value = np.sum(img_in_Gray_LR1 == max_pixel_value_LR1) / N_all_nonzero_LR1
ratio_max_pixel_value = round(ratio_max_pixel_value, 4)
print("\nRatio of the max pixel value\n>", ratio_max_pixel_value, " (", round(ratio_max_pixel_value*100, 2), "(%) )")

# Check whether the maximum pixel value is 255 in the input image(LR=1)
if max_pixel_value_LR1 == 255:
    # Calc most frequent pixel value
    img_in_Gray_nonzero_LR1         = img_in_Gray_LR1[img_in_Gray_LR1 > 0]
    bincount = np.bincount(img_in_Gray_nonzero_LR1)
    most_frequent_pixel_value_LR1   = np.argmax( bincount )
    print("\nMost frequent pixel value\n>", most_frequent_pixel_value_LR1, "(pixel value)")

    # Check whether the most frequent pixel value is 255 in the input image(LR=1)
    if most_frequent_pixel_value_LR1 == 255:
        print("\n========================================================================================")
        print("** There is a possibility that pixel value \"255\" is too much in the input image(LR=1).")
            
        # Determine standard pixel value in the input image(LR=1)
        tmp_reference_section = 0.0
        standard_pixel_value_LR1 = 254
        while tmp_reference_section < reference_section:
            # Temporarily, calc
            sum_pixels_in_section = np.sum( (standard_pixel_value_LR1 <= img_in_Gray_LR1) & (img_in_Gray_LR1 < 255) )
            tmp_reference_section = sum_pixels_in_section / N_all_nonzero_LR1

            # Next pixel value
            standard_pixel_value_LR1 -= 1

        # print("\n** final reference section")
        # print("** >", tmp_reference_section*100, "(%)")

        print("\n** Standard pixel value")
        print("** >", standard_pixel_value_LR1, "(pixel value)")

        # Calc median pixel value in the section b/w standard pixel value and maximum pixel value(255)
        section_bw_standard_255_LR1 = img_in_Gray_LR1[ (standard_pixel_value_LR1 <= img_in_Gray_LR1) & (img_in_Gray_LR1 < 255) ]
        median_bw_standard_255_LR1  = int(np.median(section_bw_standard_255_LR1))
        print("\n** Median pixel value in the section between", standard_pixel_value_LR1, "and 255")
        print("** >", median_bw_standard_255_LR1, "(pixel value)")

        # Update ratio_max_pixel_value
        ratio_old = ratio_max_pixel_value
        ratio_max_pixel_value = np.sum(img_in_Gray_LR1 == median_bw_standard_255_LR1) / N_all_nonzero_LR1
        ratio_max_pixel_value = round(ratio_max_pixel_value, 4)
        print("\n** Ratio of the pixel value", median_bw_standard_255_LR1)
        print("** >", ratio_max_pixel_value, "(", round(ratio_max_pixel_value*100, 3), "(%) )")

        print("\n** Changed ratio as follows.")
        print("** >", ratio_old, " → ", ratio_max_pixel_value)
        print("** >", round(ratio_old*100, 2), "(%) → ", round(ratio_max_pixel_value*100, 3), "(%)")

        print("========================================================================================")



# -----------------------------------------------
# ----- Correct pixel value 
#           for all images in the directory -----
# -----------------------------------------------
#image_files = glob.glob("images/serial_number_images/*.bmp")
#image_files = glob.glob("images/serial_number_images/Data_0.1t/data0/*.png")

# Set initial parameter
p_init      = 1.0
p_interval  = 0.01
p_final     = 2.0 # 0.1:1.4 0.3:1.2

img_count = 0
# for i in image_files:
#for i in range(180):
for i in range(360):
    # Read input image
    # img_in_RGB = read_img(i)
    img_in_RGB = read_img("images/serial_number_images/image{0:03d}.bmp".format(i))
    # img_in_RGB = read_img("images/serial_number_images/Data_0.1t/data0/image{0:03d}.png".format(i))

    # Then, calc number of pixels that pixel value is not 0
    img_in_Gray     = cv2.cvtColor(img_in_RGB, cv2.COLOR_RGB2GRAY)
    N_all_nonzero   = np.sum(img_in_Gray > 0)

    # Determine parameter
    # p = p_init
    # tmp_ratio_255 = 0.0
    # while tmp_ratio_255 < ratio_max_pixel_value:
    #     tmp_corrected_img_RGB = correct_pixel_value(img_in_RGB, p)
    #     tmp_corrected_img_Gray = cv2.cvtColor(tmp_corrected_img_RGB, cv2.COLOR_RGB2GRAY)

    #     # Temporarily, calc ratio of pixel value 255
    #     tmp_ratio_255 = np.sum(tmp_corrected_img_Gray == 255) / N_all_nonzero

    #     # Update parameter
    #     p += p_interval

    # p_final = round(p, 2)

    # Make output image
    img_out_RGB = correct_pixel_value(img_in_RGB, p_final)

    # Write output image
    write_img(img_out_RGB, img_count)

    # Update image count
    img_count += 1

print("\nNumber of input images\n>", img_count)
print("\n")