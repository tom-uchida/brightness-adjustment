import cv2, matplotlib
import numpy as np
import matplotlib.pyplot as plt
import sys
args = sys.argv
plt.style.use('seaborn-white')

from matplotlib import cycler
colors = cycler('color', ['#EE6666', '#3388BB', '#9988DD', '#EECC55', '#88BB44', '#FFBBBB'])
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True, grid=False, prop_cycle=colors)
plt.rc('grid', color='w', linestyle='solid')
plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=2)



# ------------------------------
# ----- Placement settings -----
# ------------------------------
fig, ax = plt.subplots(3, figsize=(9, 8)) # figsize(width, height)
fig.subplots_adjust(hspace=0.4, wspace=0.4) # interval
ax[0] = plt.subplot2grid((2,2), (0,0))
ax[1] = plt.subplot2grid((2,2), (0,1))
ax[2] = plt.subplot2grid((2,2), (1,0), colspan=2)



# ----------------------------
# ----- Read input image -----
# ----------------------------
def read_img(_img_name):
	# read input image
	img = cv2.imread(_img_name)

	# convert color (BGR → RGB)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	return img


img_1 = read_img(args[1])
img_2 = read_img(args[2])
# image information（height × width × 色数）
# print("img_origin : ", img_1.shape)  
# print("img_noised : ", img_2.shape)
# print("\n")



# ----------------------------
# ----- Show input image -----
# ----------------------------
def show_img(_i, _img, _img_name):
    ax[_i].set_title(_img_name)

    # show image
    ax[_i].imshow(_img)

    return

show_img(0, img_1,  "Input image")
show_img(1, img_2,  "Input image(LR=1)")



# -------------------------------
# ----- Convert RGB to Gray -----
# -------------------------------
img_1_gray = cv2.cvtColor(img_1, cv2.COLOR_RGB2GRAY)
img_2_gray = cv2.cvtColor(img_2, cv2.COLOR_RGB2GRAY)

# exclude pixel value == 0
img_1_gray_nonzero = img_1_gray[img_1_gray > 0]
img_2_gray_nonzero = img_2_gray[img_2_gray > 0]
print("gray_img_1 : ", img_1_gray.shape)  
print("gray_img_2 : ", img_2_gray.shape)
print("\n")



# -----------------------------------------------
# ----- Get statistical data of pixel value -----
# -----------------------------------------------
import statistics
def get_data_of_pixel_value(_pixel_values, _img_name):
    print("===== Statistical data of", _img_name ,"=====")
    print("> Max    : ", np.max(_pixel_values))
    print("> Min    : ", np.min(_pixel_values))
    print("> Mean   : ", np.mean(_pixel_values))

    # count = np.bincount(_pixel_values)
    # mode  = np.argmax(count)

    mode = statistics.mode(_pixel_values)
    print("> Most frequent value    : ", mode)
    

    print("\n")

    return 

get_data_of_pixel_value(img_1_gray_nonzero, "Input image")
get_data_of_pixel_value(img_2_gray_nonzero, "Input image (LR=1)")



# ----------------------
# ----- Matplotlib -----
# ----------------------
ax[2].hist(img_1_gray_nonzero.ravel(), bins=255, color='r', alpha=0.5, label="Input image")
ax[2].hist(img_2_gray_nonzero.ravel(), bins=255, color='b', alpha=0.5, label="Input image(LR=1)")
# R_nonzero = img_2[:,:,0][img_2[:,:,0] > 0]
# G_nonzero = img_2[:,:,1][img_2[:,:,1] > 0]
# B_nonzero = img_2[:,:,2][img_2[:,:,2] > 0]
# ax[2].hist(R_nonzero.ravel(), bins=50, color='r', alpha=0.5, label="R")
# ax[2].hist(G_nonzero.ravel(), bins=50, color='g', alpha=0.5, label="G")
# ax[2].hist(B_nonzero.ravel(), bins=50, color='b', alpha=0.5, label="B")

count_1, count_2 = np.bincount(img_1_gray_nonzero), np.bincount(img_2_gray_nonzero)
mode_1, mode_2  = np.argmax(count_1), np.argmax(count_2)
# ax[2].axvline(mode_1, color='red')
# ax[2].axvline(mode_2, color='blue')


ax[2].set_title("Comparative histograms", fontsize=12)
ax[2].set_xlabel("Pixel value", fontsize=12)
ax[2].set_ylabel("Number of pixels", fontsize=12)
ax[2].set_xlim([-10, 266])
#ax[2].set_ylim([0, 750000])
#plt.grid()
ax[2].legend(fontsize=12)

#fig.show()
plt.show()