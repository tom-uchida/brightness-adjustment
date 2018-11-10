import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
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
print("\n===== Initial parameter =====")
input_image_data = args[1]
print("input_image_data\n>",    input_image_data, "(args[1])")
print("\np_init\n>",            p_init)
print("\np_interval\n>",        p_interval)



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
    fig = plt.figure(figsize=(13,7))
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
    ax4.set_ylim([0, (max(list_rgb_max) + max(list_rgb_max)*0.05)/3])
    ax5.set_ylim([0, (max(list_rgb_max) + max(list_rgb_max)*0.05)/3])



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
img_in_RGB      = read_img(input_image_data)
img_in_LR1_RGB  = read_img(args[2])

print("\n\n===== Pre-processing =====")
print("Input image(RGB)\n>", img_in_RGB.shape) # （height, width, channel）

# Calc all number of pixels of the input image
N_all = img_in_RGB.shape[0]*img_in_RGB.shape[1]
print("\nN_all\n>", N_all, "(pixels)")

# Then, calc number of pixels that pixel value is 0
img_in_Gray = cv2.cvtColor(img_in_RGB, cv2.COLOR_RGB2GRAY)
N_all_nonzero = np.sum(img_in_Gray > 0)
print("\nN_all_nonzero\n>", N_all_nonzero, "(pixels)")

# From the input image with LR = 1, 
#   calc the ratio that the pixel is 255 after correction
img_in_LR1_gray = cv2.cvtColor(img_in_LR1_RGB, cv2.COLOR_RGB2GRAY)
img_in_LR1_gray_nonzero = img_in_LR1_gray[img_in_LR1_gray>0]
N_all_nonzero_LR1 = np.sum(img_in_LR1_gray_nonzero > 0)
ratio_overexpose = round(np.sum(img_in_LR1_gray == 255) / N_all_nonzero_LR1 * 0.01, 4)
print("\nratio_overexpose\n>",  round(ratio_overexpose*100, 2), "(%)")

# Calc the theoretical number of pixels that the pixel value is 255 after correction
N_theor = int(N_all_nonzero * ratio_overexpose)
print("\nN_theor(", round(ratio_overexpose*100, 2),"%)\n>", N_theor, "(pixels) (=",N_all_nonzero," * ",ratio_overexpose,")")



# --------------------------------
# -----  Correct pixel value -----
# --------------------------------
def correct_pixel_value(_rgb_img, _param):
    red   = cv2.multiply(_rgb_img[:, :, 0], _param) # R
    green = cv2.multiply(_rgb_img[:, :, 1], _param) # G
    blue  = cv2.multiply(_rgb_img[:, :, 2], _param) # B

    # Apply change
    corrected_img_RGB = np.empty((_rgb_img.shape[0], _rgb_img.shape[1], 3), dtype=np.uint8)
    corrected_img_RGB[:, :, 0] = red
    corrected_img_RGB[:, :, 1] = green
    corrected_img_RGB[:, :, 2] = blue

    return corrected_img_RGB



# --------------------------
# ----- Calc parameter -----
# --------------------------
p = p_init
count_equal_255 = 0
while count_equal_255 < N_theor:
    tmp_img_RGB = correct_pixel_value(img_in_RGB, p)
    tmp_img_Gray = cv2.cvtColor(tmp_img_RGB, cv2.COLOR_RGB2GRAY)

    # Count number of max pixel value(==255)
    count_equal_255 = np.sum(tmp_img_Gray == 255)
    p += p_interval

print("\n\n===== Result =====")
# Decide parameter value that meet requirement
p_final = round(p, 2)
print("p_final\n>",p_final)
print("\nNumber of pixels that pixel value is 255\n>",count_equal_255, "(pixels)")
print("\nThe ratio at which pixel value finally reached 255\n>",round(count_equal_255 / N_all_nonzero * 100, 2), "(%)")
print("\n")

# Make output image
img_out_RGB = correct_pixel_value(img_in_RGB, p_final)
# print("\nOutput image(RGB)\n>", img_out_RGB.shape) # （height × width × 色数）
# print("\n")



# -----------------------------------------
# ----- Apply tone curve with p_final -----
# -----------------------------------------
plot_tone_curve_and_histogram(tone_curve, p_final, img_in_RGB, img_out_RGB)



# ----------------------------------
# ----- Save figure and images -----
# ----------------------------------
fig_name = "images/figure_"+str(p_final)+"_"+str(round(ratio_overexpose*100,2))+".png"
plt.savefig(fig_name)
#plt.show()

# convert color (RGB → BGR)
img_in_BGR = cv2.cvtColor(img_in_RGB, cv2.COLOR_RGB2BGR)
img_out_BGR = cv2.cvtColor(img_out_RGB, cv2.COLOR_RGB2BGR)
input_img_name = "images/input.jpg"
output_img_name = "images/improved_"+str(p_final)+"_"+str(round(ratio_overexpose*100,2))+".jpg"
cv2.imwrite(input_img_name, img_in_BGR)
cv2.imwrite(output_img_name, img_out_BGR)



# -------------------------
# ----- Exec. command -----
# -------------------------
preview_command = ['code', fig_name, input_img_name, output_img_name]
# preview_command = ['open', fig_name, input_img_name, output_img_name]
try:
	res = subprocess.check_call(preview_command)

except:
	print("ERROR")