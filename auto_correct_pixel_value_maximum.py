import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cycler
import matplotlib.gridspec as gridspec
import matplotlib.patches as pat
import cv2
import subprocess
import sys
import statistics

plt.style.use('seaborn-white')

args = sys.argv
if len(args) != 3:
    raise Exception('\nUSAGE\n> $ python auto_correct_pixel_value.py [input_image_data] [input_image_data(LR=1)]')
    raise Exception('\n\nFor example\n> $ python auto_correct_pixel_value.py [input_image.bmp] [input_image_LR1.bmp]]\n')
    sys.exit()

colors = cycler('color', ['#EE6666', '#3388BB', '#9988DD', '#EECC55', '#88BB44', '#FFBBBB'])
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True, grid=False, prop_cycle=colors)
plt.rc('grid', color='w', linestyle='solid')
plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=2)



# ---------------------------------
# ----- Set initial parameter -----
# ---------------------------------
p_init = 1.0
p_interval = 0.01
print("\n===== Initial parameter =====")
print("input_image_data\n>",            args[1], "(args[1])")
print("\ninput_image_data(LR=1)\n>",    args[2], "(args[2])")
print("\np_init\n>",                    p_init)
print("\np_interval\n>",                p_interval)



# -------------------------
# ----- RGB histogram -----
# -------------------------
def rgb_hist(_img_rgb, _ax):
    R_nonzero = _img_rgb[:,:,0][_img_rgb[:,:,0] > 0]
    G_nonzero = _img_rgb[:,:,1][_img_rgb[:,:,1] > 0]
    B_nonzero = _img_rgb[:,:,2][_img_rgb[:,:,2] > 0]
    _ax.hist(R_nonzero.ravel(), bins=50, color='r', alpha=0.5, label="R")
    _ax.hist(G_nonzero.ravel(), bins=50, color='g', alpha=0.5, label="G")
    _ax.hist(B_nonzero.ravel(), bins=50, color='b', alpha=0.5, label="B")
    _ax.legend()

    _ax.set_title('RGB histogram')
    _ax.set_xlim([-5,260])
    
    return _ax



def plot_tone_curve_and_histogram(f, _p_final, _img_in_RGB, _img_out_RGB):
    fig = plt.figure(figsize=(13, 7))
    gs = gridspec.GridSpec(2,3)
    x = np.arange(256)
    
    # Tone curve
    ax2 = fig.add_subplot(gs[:,1])
    ax2.set_title('Tone Curve (parameter='+str(p_final)+')')
    ax2.set_xlabel('Input pixel value')
    ax2.set_ylabel('Output pixel value')
    ax2.set_aspect('equal')
    ax2.plot(x, f(x, _p_final), color='black')

    # Input image
    ax1 = fig.add_subplot(gs[0,0])
    ax1.set_title('Input image')
    ax1.imshow(_img_in_RGB)
    ax1.set_xticks([]), ax1.set_yticks([]) # off scale

    # Output image
    ax3 = fig.add_subplot(gs[0,2])
    ax3.set_title('Improved image')
    ax3.imshow(_img_out_RGB)
    ax3.set_xticks([]), ax3.set_yticks([])
    
    # Histogram(input image)
    ax4 = fig.add_subplot(gs[1,0])
    ax4 = rgb_hist(_img_in_RGB, ax4)

    # Histogram(output image)
    ax5 = fig.add_subplot(gs[1,2])
    ax5 = rgb_hist(_img_out_RGB, ax5)

    # Unify ylim b/w input image and improved image
    hist_in, bins_in = np.histogram(_img_in_RGB[_img_in_RGB>0], 50)
    hist_out, bins_out = np.histogram(_img_out_RGB[_img_out_RGB>0], 50)
    list_rgb_max = [max(hist_in), max(hist_out)]
    ax4.set_ylim([0, (max(list_rgb_max) + max(list_rgb_max)*0.05)/2.5])
    ax5.set_ylim([0, (max(list_rgb_max) + max(list_rgb_max)*0.05)/2.5])



# -------------------------------
# ----- Tone Curve Function -----
# -------------------------------
def tone_curve(_x, _param):
    y = np.where(_x < 255/_param, _param*_x, 255)
    return y



# ----------------------------
# ----- Read Input Image -----
# ----------------------------
def read_img(_img_name):
    # read input image
    img_BGR = cv2.imread(_img_name)

    # convert color (BGR → RGB)
    img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

    return img_RGB

# Read input image
img_in_RGB      = read_img(args[1])
img_in_RGB_LR1  = read_img(args[2])

print("\n\n===== Pre-processing =====")
print("Input image(RGB)\n>", img_in_RGB.shape) # （height, width, channel）

# Calc all number of pixels of the input image
N_all = img_in_RGB.shape[0]*img_in_RGB.shape[1]
print("\n-----", args[1], "-----")
print("N_all\n>", N_all, "(pixels)")

# Then, calc number of pixels that pixel value is not 0
img_in_Gray     = cv2.cvtColor(img_in_RGB, cv2.COLOR_RGB2GRAY)
N_all_nonzero   = np.sum(img_in_Gray > 0)
print("\nN_all_nonzero\n>", N_all_nonzero, "(pixels)")

# Calc max pixel value of the input image(LR=1)
img_in_Gray_LR1     = cv2.cvtColor(img_in_RGB_LR1, cv2.COLOR_RGB2GRAY)
N_all_nonzero_LR1   = np.sum(img_in_Gray_LR1 > 0)
max_pixel_value_LR1 = np.max(img_in_Gray_LR1)
print("\n-----", args[2], "-----")
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

        # Calc mean pixel value
        mean_LR1 = round(img_in_Gray_nonzero_LR1.mean(), 1)
        print("\n** Mean pixel value (LR=1)")
        print("** >", mean_LR1, "(pixel value)")

        # Calc median pixel value in the section b/w mean pixel value and maximum pixel value(255)
        section_bw_mean_255_LR1 = img_in_Gray_LR1[ (mean_LR1 <= img_in_Gray_LR1) & (img_in_Gray_LR1 <= 255) ]
        median_bw_mean_255_LR1  = np.median(section_bw_mean_255_LR1)
        print("\n** Median pixel value in the section between", mean_LR1, "and 255")
        print("** >", median_bw_mean_255_LR1, "(pixel value)")

        # Calc ratio 
        ratio_old = ratio_max_pixel_value
        ratio_max_pixel_value = np.sum(img_in_Gray_LR1 == median_bw_mean_255_LR1) / N_all_nonzero_LR1
        ratio_max_pixel_value = round(ratio_max_pixel_value, 4)
        print("\n** Ratio of the pixel value", median_bw_mean_255_LR1)
        print("** >", ratio_max_pixel_value, "(", ratio_max_pixel_value*100, "(%) )")

        print("\n** Changed ratio as follows.")
        print("** >", ratio_old, " → ", ratio_max_pixel_value)
        print("** >", round(ratio_old*100, 2), "(%) → ", ratio_max_pixel_value*100, "(%)")
        print("========================================================================================")



# --------------------------------------------
# ----- Correct pixel value for each RGB -----
# --------------------------------------------
def correct_pixel_value(_rgb_img, _param):
    corrected_img_RGB = np.empty((_rgb_img.shape[0], _rgb_img.shape[1], 3), dtype=np.uint8)

    # Apply correction
    corrected_img_RGB[:, :, 0] = cv2.multiply(_rgb_img[:, :, 0], _param) # R
    corrected_img_RGB[:, :, 1] = cv2.multiply(_rgb_img[:, :, 1], _param) # G
    corrected_img_RGB[:, :, 2] = cv2.multiply(_rgb_img[:, :, 2], _param) # B

    return corrected_img_RGB



# -------------------------------
# ----- Determine parameter -----
# -------------------------------
p = p_init
tmp_ratio_255 = 0.0
while tmp_ratio_255 < ratio_max_pixel_value:
    tmp_corrected_img_RGB = correct_pixel_value(img_in_RGB, p)
    tmp_corrected_img_Gray = cv2.cvtColor(tmp_corrected_img_RGB, cv2.COLOR_RGB2GRAY)

    # Temporarily, calc ratio of pixel value 255
    tmp_ratio_255 = np.sum(tmp_corrected_img_Gray == 255) / N_all_nonzero

    # Update parameter
    p += p_interval

p_final = round(p, 2)

# Make output image
img_out_RGB  = correct_pixel_value(img_in_RGB, p_final)
img_out_Gray = cv2.cvtColor(img_out_RGB, cv2.COLOR_RGB2GRAY)

print("\n\n===== Result =====")
print("p_final\n>", p_final)
print("\nThe ratio at which pixel value finally reached 255\n>", round(np.sum(img_out_Gray==255) / N_all_nonzero * 100, 2), "(%)")
print("\n")



# -----------------------------------------
# ----- Apply tone curve with p_final -----
# -----------------------------------------
plot_tone_curve_and_histogram(tone_curve, p_final, img_in_RGB, img_out_RGB)



# ----------------------------------
# ----- Save figure and images -----
# ----------------------------------
fig_name = "images/figure_"+str(p_final)+".png"
plt.savefig(fig_name)
# plt.show()

# convert color (RGB → BGR)
img_in_BGR = cv2.cvtColor(img_in_RGB, cv2.COLOR_RGB2BGR)
img_out_BGR = cv2.cvtColor(img_out_RGB, cv2.COLOR_RGB2BGR)
input_img_name = "images/input.jpg"
output_img_name = "images/improved_"+str(p_final)+".jpg"
cv2.imwrite(input_img_name, img_in_BGR)
cv2.imwrite(output_img_name, img_out_BGR)



# -------------------------
# ----- Exec. command -----
# -------------------------
# preview_command = ['code', fig_name, input_img_name, output_img_name]
# # preview_command = ['open', fig_name, input_img_name, output_img_name]
# try:
# 	res = subprocess.check_call(preview_command)

# except:
# 	print("ERROR")