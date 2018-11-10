import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import subprocess
import sys
args = sys.argv
if len(args) != 6:
    raise Exception('\nUSAGE\n> $ python auto_correct_pixel_value.py [input_image_data] [ratio_for_low] [ratio_for_high] [p_init] [ratio_for_high] [p_interval]')
    raise Exception('\n\nFor example\n> $ python auto_correct_pixel_value.py input_image.jpg 2 0.001 0.01\n')
    sys.exit()


plt.style.use('seaborn-white')

from matplotlib import cycler
colors = cycler('color', ['#EE6666', '#3388BB', '#9988DD', '#EECC55', '#88BB44', '#FFBBBB'])
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True, grid=False, prop_cycle=colors)
plt.rc('grid', color='w', linestyle='solid')
plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=2)



# ---------------------------------
# ----- Set initial parameter -----
# ---------------------------------
print("\n===== Initial parameter =====")
input_image_data    = args[1]
ratio_for_low       = float(args[2])
ratio_for_high      = float(args[3])
p_init              = float(args[4])
p_interval          = float(args[5])
print("input_image_data\n>",input_image_data,"(args[1])")
print("\nratio_for_low\n>",ratio_for_low,"(args[2])")
print("\nratio_for_high\n>",ratio_for_high,"(args[3])")
print("\np_init\n>",p_init,"(args[4])")
print("\np_interval\n>",p_interval,"(args[5])")



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
    ax2.set_title('Tone Curve (parameter='+str(_p_final)+')')
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
    ax3.set_title('Output image')
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

    fig_name = "images/figure_"+str(_p_final)+".png"
    plt.savefig(fig_name)



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
img_in_RGB = read_img(input_image_data)

print("\n\n\n===== Pre-processing =====")
print("Input image(RGB)\n>", img_in_RGB.shape) # (height, width, channel)

# Calc all number of pixels of the input image
N_all = img_in_RGB.shape[0]*img_in_RGB.shape[1]
print("\nN_all\n>", N_all, "(pixels)")

# Then, calc number of pixels that exclude backgroung color
img_in_Gray = cv2.cvtColor(img_in_RGB, cv2.COLOR_RGB2GRAY)
N_all_nonzero = np.sum(img_in_Gray > 0)
print("\nN_all_nonzero\n>", N_all_nonzero, "(pixels)")

# Calc mean pixel value of the input image
img_in_Gray_nonzero = img_in_Gray[img_in_Gray > 0]
mean = round(img_in_Gray_nonzero.mean(), 1)
print("\nMean of pixel values of the input image\n>",mean, "(pixel value)")

# Calc SD pixel value of the input image
SD = round(img_in_Gray_nonzero.std(), 1)
print("\nStandard Deviation of pixel values of the input image\n>",SD, "(pixel value)")



# ---------------------------------------------------------
# ----- Decompose the input image into two images 
#           with the mean pixel value as the boundary -----
# ---------------------------------------------------------
low_num_nonzero  = np.count_nonzero( (img_in_Gray <= mean) & (img_in_Gray > 0) )
low_num  = np.count_nonzero(  img_in_Gray <= mean )
high_num = np.count_nonzero(  img_in_Gray >  mean )
print("\nNumber of \"low\" pixel value\n>", low_num_nonzero, "(pixels) out of", N_all_nonzero, "(pixels)")
print(">", round(low_num_nonzero/N_all_nonzero*100,1), "(%)")
print("\nNumber of \"high\" pixel value\n>", high_num, "(pixels) out of", N_all_nonzero, "(pixels)")
print(">", round(high_num/N_all_nonzero*100,1), "(%)")

# ndarray dtype:bool
#low_index_bool  = (img_in_Gray <= mean) & (img_in_Gray > 0) 
low_index_bool  =  img_in_Gray <= mean
#high_index_bool =  img_in_Gray >  mean
high_index_bool =  ~low_index_bool

# Decompose the input image into R,G,B channel
img_in_R, img_in_G, img_in_B = img_in_RGB[:,:,0], img_in_RGB[:,:,1], img_in_RGB[:,:,2]

# Apply decomposition
low_R  = np.where(low_index_bool,  img_in_R, 0)
low_G  = np.where(low_index_bool,  img_in_G, 0)
low_B  = np.where(low_index_bool,  img_in_B, 0)
high_R = np.where(high_index_bool, img_in_R, 0)
high_G = np.where(high_index_bool, img_in_G, 0)
high_B = np.where(high_index_bool, img_in_B, 0)

low_img_in_RGB, high_img_in_RGB = img_in_RGB.copy(), img_in_RGB.copy()
low_img_in_RGB[:,:,0],  low_img_in_RGB[:,:,1],  low_img_in_RGB[:,:,2]  = low_R,  low_G,  low_B
high_img_in_RGB[:,:,0], high_img_in_RGB[:,:,1], high_img_in_RGB[:,:,2] = high_R, high_G, high_B

# Calc the theoretical number of pixels at which finally reach 255
N_theor_low = int(low_num_nonzero * ratio_for_low)
print("\nN_theor_low(",ratio_for_low*100,"%)\n>", N_theor_low, "(pixels) (=",low_num_nonzero," * ",ratio_for_low,")")

N_theor_high = int(high_num * ratio_for_high)
print("\nN_theor_high(",ratio_for_high*100,"%)\n>", N_theor_high, "(pixels) (=",high_num," * ",ratio_for_high,")")



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



# ----------------------------------
# ----- Calc parameter for low -----
# ----------------------------------
p = p_init
count_equal_255 = 0
while count_equal_255 < N_theor_low:
    tmp_img_RGB = correct_pixel_value(low_img_in_RGB, p)
    tmp_img_Gray = cv2.cvtColor(tmp_img_RGB, cv2.COLOR_RGB2GRAY)

    # Count number of max pixel value(==255)
    count_equal_255 = np.sum(tmp_img_Gray >= mean+SD)
    p += p_interval

print("\n\n\n===== Result for \"low pixel value image\" =====")
# Decide parameter value that meet requirement
p_final_low = round(p, 2)
print("p_final_low\n>",p_final_low)
print("\nNumber of pixels that pixel value is 255\n>",count_equal_255, "(pixels)")
print("\nThe ratio at which pixel value finally reached 255\n>",round(count_equal_255 / low_num_nonzero * 100, 2), "(%)")

# Make low output image
low_img_out_RGB = correct_pixel_value(low_img_in_RGB, p_final_low)



# -----------------------------------
# ----- Calc parameter for high -----
# -----------------------------------
p = p_init
count_equal_255 = 0
while count_equal_255 < N_theor_high:
    tmp_img_RGB = correct_pixel_value(high_img_in_RGB, p)
    tmp_img_Gray = cv2.cvtColor(tmp_img_RGB, cv2.COLOR_RGB2GRAY)

    # Count number of max pixel value(==255)
    count_equal_255 = np.sum(tmp_img_Gray == 255)
    p += p_interval

print("\n\n\n===== Result for \"high pixel value image\" =====")
# Decide parameter value that meet requirement
p_final_high = round(p, 2)
print("p_final_high\n>",p_final_high)
print("\nNumber of pixels that pixel value is 255\n>",count_equal_255, "(pixels)")
print("\nThe ratio at which pixel value finally reached 255\n>",round(count_equal_255 / high_num * 100, 2), "(%)")
print("\n")

# Make high output image
high_img_out_RGB = correct_pixel_value(high_img_in_RGB, p_final_high)



# ------------------------------------------------------
# ----- Synthesize low and high pixel value images -----
# ------------------------------------------------------
img_out_RGB = cv2.scaleAdd(low_img_out_RGB, 1.0, high_img_out_RGB)



# -----------------------------------------
# ----- Apply tone curve with p_final -----
# -----------------------------------------
plot_tone_curve_and_histogram(tone_curve, p_final_low, low_img_in_RGB, low_img_out_RGB)
plot_tone_curve_and_histogram(tone_curve, p_final_high, high_img_in_RGB, high_img_out_RGB)



# ----------------------------------
# ----- Save figure and images -----
# ----------------------------------
#plt.show()

# Convert color (RGB → BGR)
img_in_BGR = cv2.cvtColor(img_in_RGB, cv2.COLOR_RGB2BGR)
img_out_BGR_low  = cv2.cvtColor(low_img_out_RGB, cv2.COLOR_RGB2BGR)
img_out_BGR_high = cv2.cvtColor(high_img_out_RGB, cv2.COLOR_RGB2BGR)
img_out_BGR = cv2.cvtColor(img_out_RGB, cv2.COLOR_RGB2BGR)
input_img_name = "images/input.jpg"
low_output_img_name  = "images/low_improved_"+str(p_final_low)+"_"+str(ratio_for_low)+".jpg"
high_output_img_name = "images/high_improved_"+str(p_final_high)+"_"+str(ratio_for_high)+".jpg"
output_img_name = "images/improved_"+str(p_final_low)+"_"+str(p_final_high)+".jpg"
cv2.imwrite(input_img_name, img_in_BGR)
cv2.imwrite(low_output_img_name, img_out_BGR_low)
cv2.imwrite(high_output_img_name, img_out_BGR_high)
cv2.imwrite(output_img_name, img_out_BGR)

# Save low and high images
low_img_in_BGR = cv2.cvtColor(low_img_in_RGB, cv2.COLOR_RGB2BGR)
high_img_in_BGR = cv2.cvtColor(high_img_in_RGB, cv2.COLOR_RGB2BGR)
low_img_name = "images/low.jpg"
high_img_name = "images/high.jpg"
cv2.imwrite(low_img_name, low_img_in_BGR)
cv2.imwrite(high_img_name, high_img_in_BGR)



# -------------------------
# ----- Exec. command -----
# -------------------------
# preview_command = ['code', fig_name, input_img_name, output_img_name]
# # preview_command = ['open', fig_name, input_img_name, output_img_name]
# try:
# 	res = subprocess.check_call(preview_command)

# except:
# 	print("ERROR")