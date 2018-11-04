import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import subprocess
import sys
args = sys.argv
#print(len(args))
if len(args) != 5:
    raise Exception('\nUSAGE\n> $ python auto_correct_pixel_value.py [input_image_data] [p_init] [ratio] [p_interval]')
    raise Exception('\n\nFor example\n> $ python auto_correct_pixel_value.py input_image.jpg 2 0.005 0.01\n')
    sys.exit()

#import seaborn as sns
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
p_init              = float(args[2])    # 2
ratio               = float(args[3])    # 0.005
p_interval          = float(args[4])  # 0.01
print("input_image_data\n>",input_image_data,"(args[1])")
print("\np_init\n>",p_init,"(args[2])")
print("\nratio\n>",ratio,"(args[3])")
print("\np_interval\n>",p_interval,"(args[4])")



# -------------------------
# ----- RGB histogram -----
# -------------------------
def rgb_hist(rgb_img, ax, ticks=None):
    if ticks:
        ax.set_xticks(ticks)
    
    R_nonzero = rgb_img[:,:,0][rgb_img[:,:,0] > 0]
    G_nonzero = rgb_img[:,:,1][rgb_img[:,:,1] > 0]
    B_nonzero = rgb_img[:,:,2][rgb_img[:,:,2] > 0]
    ax.hist(R_nonzero.ravel(), bins=50, color='r', alpha=0.5, label="R")
    ax.hist(G_nonzero.ravel(), bins=50, color='g', alpha=0.5, label="G")
    ax.hist(B_nonzero.ravel(), bins=50, color='b', alpha=0.5, label="B")
    ax.legend()

    ax.set_title('RGB histogram')
    ax.set_xlim([-5,260])
    ax.set_ylim([0,30000])
    
    return ax



def plot_curve_and_histogram(f, _p_final, _img_in_RGB, _img_out_RGB):
    fig = plt.figure(figsize=(13,7))
    gs = gridspec.GridSpec(2,3)
    x = np.arange(256)
    
    # Tone curve
    ax2 = fig.add_subplot(gs[:,1]) # 2列目
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
    ax3.set_title('Output image')
    ax3.imshow(_img_out_RGB)
    ax3.set_xticks([]), ax3.set_yticks([])
    
    # Histogram(input image)
    ax4 = fig.add_subplot(gs[1,0])
    ax4 = rgb_hist(_img_in_RGB, ax4)

    # Histogram(output image)
    ax5 = fig.add_subplot(gs[1,2])
    ax5 = rgb_hist(_img_out_RGB, ax5)



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

# Select input image
# img_in_RGB = read_img("images/2018-09-30/out_gaussian_RL100.jpg")
# img_in_RGB = read_img("images/2018-10-01/funehoko200.jpg")
# img_in_RGB = read_img("images/2018-10-19/RL100_3072.bmp")
# img_in_RGB = read_img("images/2018-10-29/DATA/20160724_RL10.bmp")
# img_in_RGB = read_img("images/2018-10-29/DATA/nakajimake_RL100.bmp")
# img_in_RGB = read_img("images/2018-10-29/DATA/zuiganzi_RL100.bmp")
img_in_RGB = read_img(input_image_data)

print("\n\n===== Pre-Process =====")
print("Input image(RGB)\n>", img_in_RGB.shape) # （height × width × 色数）

# Calc number of pixels
N_all = img_in_RGB.shape[0]*img_in_RGB.shape[1]
print("\nN_all\n>", N_all, "(pixels)")

# Calc number of pixels(exclude backgroung color)
img_in_Gray = cv2.cvtColor(img_in_RGB, cv2.COLOR_RGB2GRAY)
N_all_nonzero = np.sum(img_in_Gray > 0)
print("\nN_all_nonzero\n>", N_all_nonzero, "(pixels)")

# Calc number of pixels
N_theor = int(N_all_nonzero * ratio)
print("\nN_theor(",ratio*100,"%)\n>", N_theor, "(pixels) (=",N_all_nonzero," * ",ratio,")")



# --------------------------------
# -----  Correct pixel value -----
# --------------------------------
def correct_pixel_value(_rgb_img, _param):
  red   = cv2.multiply(_rgb_img[:, :, 0], _param) # R
  green = cv2.multiply(_rgb_img[:, :, 1], _param) # G
  blue  = cv2.multiply(_rgb_img[:, :, 2], _param) # B

  # Apply change
  changed_img_RGB = np.empty((_rgb_img.shape[0], _rgb_img.shape[1], 3), dtype=np.uint8)
  changed_img_RGB[:, :, 0] = red
  changed_img_RGB[:, :, 1] = green
  changed_img_RGB[:, :, 2] = blue

  return changed_img_RGB



# --------------------------
# ----- Calc parameter -----
# --------------------------
count_equal_255 = 0
while count_equal_255 < N_theor:
    tmp_img_RGB = correct_pixel_value(img_in_RGB, p_init)
    tmp_img_Gray = cv2.cvtColor(tmp_img_RGB, cv2.COLOR_RGB2GRAY)

    # Count number of max pixel value(==255)
    count_equal_255 = np.sum(tmp_img_Gray == 255)
    p_init += p_interval

print("\n\n===== Result =====")
# Decide parameter value that meet requirement
p_final = round(p_init, 2)
print("p_final\n>", p_final)
print("\nNumber of pixels that pixel value is 255\n>",count_equal_255, "(pixels)")
print("\nThe ratio at which pixel value finally reached 255\n>",count_equal_255 / N_all_nonzero, "(%)")
print("\n")

# Make output image
img_out_RGB = correct_pixel_value(img_in_RGB, p_final)
# print("\nOutput image(RGB)\n>", img_out_RGB.shape) # （height × width × 色数）
# print("\n")

# Execute
plot_curve_and_histogram(tone_curve, p_final, img_in_RGB, img_out_RGB)
    


# ----------------------------------
# ----- Save figure and images -----
# ----------------------------------
fig_name = "images/figure_"+str(p_final)+"_"+str(ratio)+".png"
plt.savefig(fig_name)
#plt.show()

# convert color (RGB → BGR)
img_in_BGR = cv2.cvtColor(img_in_RGB, cv2.COLOR_RGB2BGR)
img_out_BGR = cv2.cvtColor(img_out_RGB, cv2.COLOR_RGB2BGR)
input_img_name = "images/input.jpg"
output_img_name = "images/improved_"+str(p_final)+"_"+str(ratio)+".jpg"
cv2.imwrite(input_img_name, img_in_BGR)
cv2.imwrite(output_img_name, img_out_BGR)



# -------------------------
# ----- Exec. command -----
# -------------------------
preview_command = ['open', fig_name, input_img_name, output_img_name]
try:
	res = subprocess.check_call(preview_command)

except:
	print("ERROR")