import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import sys
args = sys.argv
#print(len(args))
if len(args) != 4:
    raise Exception('\nUSAGE\n> $ tone_curve.py [p_init] [ratio] [p_update]\n\nFor example\n> $ tone_curve.py 2 0.005 0.01\n')
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
p_init      = float(args[1])    # 2
ratio       = float(args[2])    # 0.005
p_update    = float(args[3])    # 0.01
print("p_init\n>",p_init,"(args[1])")
print("\nratio\n>",ratio,"(args[2])")
print("\np_update\n>",p_update,"(args[3])")



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
    ax.set_ylim([0,41000])
    
    return ax



def plot_curve(f, _p_final, _img_in_RGB, _img_out_RGB):
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
    
    plt.show()

    # save image
    # convert color (RGB → BGR)
    img_out_BGR = cv2.cvtColor(_img_out_RGB, cv2.COLOR_RGB2BGR)
    cv2.imwrite("images/output_improved_"+str(p_final)+"_"+str(ratio)+".jpg", img_out_BGR)



# -------------------------------
# ----- Tone Curve Function -----
# -------------------------------
def curve_1(_x):
    y = 3*_x
    return y

def curve_2(_x):
    y = (np.sin(np.pi * (_x/255 - 0.5)) + 1)/2 * 255
    return y

def curve_gamma(_x, _param):
    gamma = 1.5 # 3,2,1/2
    y = 255*(_x/255)**(1/gamma)
    return y

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
img_in_RGB = read_img("images/2018-10-19/RL100_3072.bmp")
print("\n\n===== Pre-Process =====")
print("Input image(RGB)\n>", img_in_RGB.shape) # （height × width × 色数）

# Calc number of pixels
N_all = img_in_RGB.shape[0]*img_in_RGB.shape[1]
print("\nN_all\n>", N_all, "(pixels)")

# Calc number of pixels(exclude backgroung color)
img_in_Gray = cv2.cvtColor(img_in_RGB, cv2.COLOR_RGB2GRAY)
N_all_nonzero = np.sum(img_in_Gray > 0)
print("\nN_all_nonzero\n>", N_all_nonzero, "(pixels)")

# Calc  number of pixels
N_meet = int(N_all_nonzero * ratio)
print("\nN_meet(",ratio*100,"%)\n>", N_meet, "(pixels) (=",N_all_nonzero," * ",ratio,")")



# -----------------------------
# ----- Change color tone -----
# -----------------------------
def change_color_tone(_rgb_img, _param):
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
while count_equal_255 < N_meet:
    tmp_img_RGB = change_color_tone(img_in_RGB, p_init)
    tmp_img_Gray = cv2.cvtColor(tmp_img_RGB, cv2.COLOR_RGB2GRAY)

    # Count number of max pixel value(==255)
    count_equal_255 = np.sum(tmp_img_Gray == 255)
    p_init += p_update

print("\n\n===== Result =====")
# Decide parameter value that meet requirement
p_final = round(p_init, 2)
print("p_final\n>", p_final)

# Print number of max pixel value(==255)
print("\nCount number of max pixel value(pixel value:255)\n>",count_equal_255, "(pixels)")

# Make output image
img_out_RGB = change_color_tone(img_in_RGB, p_final)
print("\nOutput image(RGB)\n>", img_out_RGB.shape) # （height × width × 色数）



# ---------------------
# ----- Execution -----
# ---------------------
plot_curve(tone_curve, p_final, img_in_RGB, img_out_RGB)