import cv2
import numpy as np
import sys
from matplotlib import cycler
import matplotlib.patches as pat
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

# Graph settings
# plt.style.use('seaborn-white')
plt.style.use('bmh')
colors = cycler('color', ['#EE6666', '#3388BB', '#9988DD', '#EECC55', '#88BB44', '#FFBBBB'])
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True, grid=False, prop_cycle=colors)
# plt.rc('grid', color='w', linestyle='solid')
# plt.rc('patch', edgecolor='#E6E6E6')
# plt.rc('lines', linewidth=2)
# plt.rcParams['font.family'] = 'IPAGothic' # font setting
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["mathtext.rm"] = "Times New Roman"

# Check arguments
args = sys.argv
if len(args) != 2:
    #raise Exception
    sys.exit()

def read_image(_img_name):
    img_BGR = cv2.imread(_img_name)
    img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

    return img_RGB
# End of read_image()

def create_RGB_hist(_img_RGB, _ax, _title):
    tmp_b_idx_bgcolor = (_img_RGB[:,:,0]==BGColor[0]) & (_img_RGB[:,:,1]==BGColor[1]) & (_img_RGB[:,:,2]==BGColor[2])
    img_R_non_bgcolor = _img_RGB[:,:,0][~tmp_b_idx_bgcolor]
    img_G_non_bgcolor = _img_RGB[:,:,1][~tmp_b_idx_bgcolor]
    img_B_non_bgcolor = _img_RGB[:,:,2][~tmp_b_idx_bgcolor]
    _ax.hist(img_R_non_bgcolor.ravel(), bins=bin_number, color='r', alpha=0.5, label="R")
    _ax.hist(img_G_non_bgcolor.ravel(), bins=bin_number, color='g', alpha=0.5, label="G")
    _ax.hist(img_B_non_bgcolor.ravel(), bins=bin_number, color='b', alpha=0.5, label="B")
    _ax.legend()

    _ax.set_title(_title, fontsize='14')
    _ax.set_xlim([-5, 260])
    
    return _ax

def create_figure(_fig_name, _threshold):
    # Create figure
    fig = plt.figure(figsize=(6, 8)) # figsize=(width, height)
    gs  = gridspec.GridSpec(2,1)

    # Input image
    ax_img_in = fig.add_subplot(gs[0,0])
    ax_img_in.set_title("Input image", fontsize='14')
    ax_img_in.imshow(img_in_RGB)
    ax_img_in.set_xticks([]), ax_img_in.set_yticks([])

    # Histogram of the input image
    ax_hist_in = fig.add_subplot(gs[1,0])
    ax_hist_in = create_RGB_hist(img_in_RGB, ax_hist_in, "Histogram")
    ax_hist_in.set_xlim([-5, 260])

    # Draw line
    ax_hist_in.axvline(_threshold, color='black')

    plt.savefig(_fig_name)

if __name__ == "__main__":
    # Read an input image
    img_in_RGB = read_image(args[1])

    # Convert RGB to Grayscale
    img_in_Gray = cv2.cvtColor(img_in_RGB, cv2.COLOR_RGB2GRAY)

    # Otsu method
    threshold, img_out_Gray_otsu = cv2.threshold(img_in_Gray, 0, 255, cv2.THRESH_OTSU)

    print("Threshold pixel value: ", threshold)

    BGColor = [0, 0, 0] # Background color
    bin_number = 255
    create_figure("figure.png", threshold)

    # # Write an output image
    # cv2.imwrite("out_otsu_method.png", img_out_Gray_otsu)