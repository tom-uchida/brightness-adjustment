#
# 入力画像，入力画像（LR=1），補正画像の3枚の画像のヒストグラムを比較するプログラム
#

import cv2, matplotlib
import numpy as np
import matplotlib.pyplot as plt
import sys
args = sys.argv
plt.style.use('seaborn-white')

from matplotlib import cycler
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True, grid=False)
# plt.rc('grid', color='w', linestyle='solid')
# plt.rc('patch', edgecolor='#E6E6E6')
# plt.rc('lines', linewidth=2)



# ------------------------------
# ----- Placement settings -----
# ------------------------------
fig, ax = plt.subplots(4, figsize=(10, 8)) # figsize(width, height)
fig.subplots_adjust(hspace=0.4, wspace=0.4) # interval
ax[0] = plt.subplot2grid((2,3), (0,0))
ax[1] = plt.subplot2grid((2,3), (0,1))
ax[2] = plt.subplot2grid((2,3), (0,2))
ax[3] = plt.subplot2grid((2,3), (1,0), colspan=3)



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
img_3 = read_img(args[3])



# ----------------------------
# ----- Show input image -----
# ----------------------------
def show_img(_i, _img, _img_name):
    ax[_i].set_title(_img_name)

    # show image
    ax[_i].imshow(_img)

    return

show_img(0, img_1,  "Input image")
show_img(1, img_2,  "Improved image")
show_img(2, img_3,  "Input image (LR=1)")



# -------------------------------
# ----- Convert RGB to Gray -----
# -------------------------------
img_1_gray = cv2.cvtColor(img_1, cv2.COLOR_RGB2GRAY)
img_2_gray = cv2.cvtColor(img_2, cv2.COLOR_RGB2GRAY)
img_3_gray = cv2.cvtColor(img_3, cv2.COLOR_RGB2GRAY)

img_1_gray_nonzero = img_1_gray[img_1_gray > 0]
img_2_gray_nonzero = img_2_gray[img_2_gray > 0]
img_3_gray_nonzero = img_3_gray[img_3_gray > 0]



# -----------------------------------------------
# ----- Get statistical data of pixel value -----
# -----------------------------------------------
def get_data_of_pixel_value(_pixel_values):
    print("===== Statistical Data of Pixel Values =====")
    print("> Max    : ", np.max(_pixel_values))
    print("> Min    : ", np.min(_pixel_values))
    print("> Mean   : ", np.mean(_pixel_values))
    #print("> Median : ", np.median(_pixel_values))
    print("\n")

    return

get_data_of_pixel_value(img_1_gray_nonzero)
get_data_of_pixel_value(img_2_gray_nonzero)
get_data_of_pixel_value(img_3_gray_nonzero)

mean_1 = int(np.mean(img_1_gray_nonzero))
mean_2 = int(np.mean(img_2_gray_nonzero))
mean_3 = int(np.mean(img_3_gray_nonzero))



# ----------------------
# ----- Matplotlib -----
# ----------------------
# input image
ax[3].hist(img_1_gray_nonzero.ravel(), bins=255, alpha=0.8, label="Input image")
ax[3].axvline(mean_1, color='#1F77B4')
ax[3].text(mean_1/265+0.03, 0.7, "mean:"+str(mean_1), color='#1F77B4', transform=ax[3].transAxes, fontsize=12)

# improved image
ax[3].hist(img_2_gray_nonzero.ravel(), bins=255, alpha=0.8, label="Improved image")
ax[3].axvline(mean_2, color='#FF7E0F')
ax[3].text(mean_2/265+0.03, 0.7, "mean:"+str(mean_2), color='#FF7E0F', transform=ax[3].transAxes, fontsize=12)

# input image(LR=1)
ax[3].hist(img_3_gray_nonzero.ravel(), bins=255, alpha=0.8, label="Input image (LR=1)")
ax[3].axvline(mean_3, color='#2C9F2C')
ax[3].text(mean_3/265+0.03, 0.7, "mean:"+str(mean_3), color='#2C9F2C', transform=ax[3].transAxes, fontsize=12)


ax[3].set_title("Comparative histograms", fontsize=12)
ax[3].set_xlabel("Pixel value", fontsize=12)
ax[3].set_ylabel("Number of pixels", fontsize=12)
ax[3].set_xlim([-5, 260])

ax[3].legend(fontsize=12)

#fig.show()
plt.show()