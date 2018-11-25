import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as pat
plt.style.use('seaborn-white')
import cv2
import subprocess
import sys
args = sys.argv
if len(args) != 3:
    raise Exception('\nUSAGE\n> $ python auto_correct_pixel_value.py [input_image_data] [input_image_data(LR=1)]')
    raise Exception('\n\nFor example\n> $ python auto_correct_pixel_value.py [input_image.bmp] [input_image_LR1.bmp]]\n')
    sys.exit()

from matplotlib import cycler
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
specified_section_ratio = 0.9
print("\n===== Initial parameter =====")
print("input_image_data\n>",            args[1], "(args[1])")
print("\ninput_image_data(LR=1)\n>",    args[2], "(args[2])")
print("\np_init\n>",                    p_init)
print("\np_interval\n>",                p_interval)
print("\nspecified_section_ratio\n>",   specified_section_ratio*100, "(%)")



# -------------------------
# ----- RGB histogram -----
# -------------------------
def rgb_hist(_img_rgb, _ax):  
    # Draw pixel value histogram
    # _img_gray = cv2.cvtColor(_img_rgb, cv2.COLOR_RGB2GRAY)
    # _img_gray_nonzero = _img_gray[img_in_Gray > 0]
    # _ax.hist(_img_gray_nonzero.ravel(), bins=255, color='black', alpha=1.0, label="Pixel value")

    # Draw RGB histogram
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

    # Draw line
    #ax5.axvline(standard_pixel_value, color='black')
    x_section = standard_pixel_value/265 + (5/265)
    #x_text      = x_section + 0.5*(1.0-x_section+(10/265))
    ax5.text(x_section, 0.7, str(standard_pixel_value) + " ~ " + str(np.max(img_out_Gray)), transform=ax5.transAxes, color='black')
    ax5.text(x_section, 0.6, "→ " + str(specified_section_ratio*100) + " (%)", transform=ax5.transAxes, color='black')

    # Draw rectangle
    rect = plt.Rectangle((x_section, 0.0), 1.0-x_section-(5/265), 1.0, transform=ax5.transAxes, fc='black', alpha=0.2)
    ax5.add_patch(rect)



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
print("\nN_all\n>", N_all, "(pixels)")

# Then, calc number of pixels that pixel value is 0
img_in_Gray     = cv2.cvtColor(img_in_RGB, cv2.COLOR_RGB2GRAY)
N_all_nonzero   = np.sum(img_in_Gray > 0)
print("\nN_all_nonzero\n>", N_all_nonzero, "(pixels)")

# Calc max pixel value of the input image(LR=1)
img_in_Gray_LR1     = cv2.cvtColor(img_in_RGB_LR1, cv2.COLOR_RGB2GRAY)
N_all_nonzero_LR1   = np.sum(img_in_Gray_LR1 > 0)
max_pixel_value_LR1 = np.max(img_in_Gray_LR1)
print("\nmax_pixel_value (LR=1)\n>", max_pixel_value_LR1, "(pixel value)")



# -----------------------------------------------------------------------
# ----- Search for pixel value that determines the specified section -----
# -----------------------------------------------------------------------
target_pixel_value  = max_pixel_value_LR1
tmp_ratio_LR1       = 0.0
while tmp_ratio_LR1 < specified_section_ratio:
    tmp_sum_pixel_number = np.sum( target_pixel_value <= img_in_Gray_LR1 )
    # tmp_sum_pixel_number = np.sum( (target_pixel_value <= img_in_Gray_LR1) & (img_in_Gray_LR1 < 255) )

    # Temporarily, calc specified section ratio
    tmp_ratio_LR1 = tmp_sum_pixel_number / N_all_nonzero_LR1

    # Next pixel value
    target_pixel_value -= 1

print("\n\n** Specified section was confirmed.")
standard_pixel_value = target_pixel_value
print("standard_pixel_value (LR=1)\n>", standard_pixel_value, "(pixel value)")

specified_section_ratio_LR1_final = tmp_ratio_LR1
print("\nspecified_section_ratio_LR1_final (LR=1)\n>", round(specified_section_ratio_LR1_final*100, 1), "(%) ( >=", standard_pixel_value, ")")

# 区間内ヒストグラム&統計値を計算する場合はここ．



# --------------------------------------------
# ----- Correct pixel value for each RGB -----
# --------------------------------------------
def correct_pixel_value(_rgb_img, _param):
    # Multiply
    red   = cv2.multiply(_rgb_img[:, :, 0], _param) # R
    green = cv2.multiply(_rgb_img[:, :, 1], _param) # G
    blue  = cv2.multiply(_rgb_img[:, :, 2], _param) # B

    # Add
    # red   = cv2.add(_rgb_img[:, :, 0], _param) # R
    # green = cv2.add(_rgb_img[:, :, 1], _param) # G
    # blue  = cv2.add(_rgb_img[:, :, 2], _param) # B

    # Apply correction
    corrected_img_RGB = np.empty((_rgb_img.shape[0], _rgb_img.shape[1], 3), dtype=np.uint8)
    corrected_img_RGB[:, :, 0] = red
    corrected_img_RGB[:, :, 1] = green
    corrected_img_RGB[:, :, 2] = blue

    return corrected_img_RGB



# -------------------------------
# ----- Determine parameter -----
# -------------------------------
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
img_out_RGB     = correct_pixel_value(img_in_RGB, p_final)
img_out_Gray    = cv2.cvtColor(img_out_RGB, cv2.COLOR_RGB2GRAY)
# out_N_all_nonzero   = np.sum(img_out_Gray > 0)
# print("\nout_N_all_nonzero\n>", out_N_all_nonzero, "(pixels)")

print("\n\n===== Result =====")
print("p_final\n>",p_final)
specified_section_ratio_final = tmp_ratio
print("\nspecified_section_ratio_final\n>", round(specified_section_ratio_final*100, 1), "(%) ( >=", standard_pixel_value,")")
#print("\nNumber of pixels that pixel value is 255\n>", np.sum(img_out_Gray==255), "(pixels)")
print("\nThe ratio at which pixel value finally reached 255\n>", round(np.sum(img_out_Gray==255) / N_all_nonzero * 100, 2), "(%)")
print("\n")



# -----------------------------------------
# ----- Apply tone curve with p_final -----
# -----------------------------------------
plot_tone_curve_and_histogram(tone_curve, p_final, img_in_RGB, img_out_RGB)



# ----------------------------------
# ----- Save figure and images -----
# ----------------------------------
fig_name = "images/figure_"+str(p_final)+"_"+str(round(specified_section_ratio, 2))+".png"
plt.savefig(fig_name)
# plt.show()

# convert color (RGB → BGR)
img_in_BGR = cv2.cvtColor(img_in_RGB, cv2.COLOR_RGB2BGR)
img_out_BGR = cv2.cvtColor(img_out_RGB, cv2.COLOR_RGB2BGR)
input_img_name = "images/input.jpg"
output_img_name = "images/improved_"+str(p_final)+"_"+str(round(specified_section_ratio, 2))+".jpg"
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