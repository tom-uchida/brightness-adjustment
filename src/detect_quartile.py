#
# 四分位数を検出するプログラム
#

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
show_img(1, img_2,  "Input image (LR=1)")



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
from scipy import stats
def get_data_of_pixel_value(_pixel_values):
    print("===== Statistical Data of Pixel Values =====")

    #最大値
    print ("Max     : ", np.max(_pixel_values))

    #最小値
    print ("Min     : ", np.min(_pixel_values))

    #平均値
    mean = np.mean(_pixel_values)
    print ("Mean    : ", mean)
    
    #第1四分位
    first_quater = stats.scoreatpercentile(_pixel_values, 25)
    print ("1Q      : ", first_quater)

    #中央値
    median = np.median(_pixel_values)
    print ("Median  : ", median)

    #第3四分位
    third_quater = stats.scoreatpercentile(_pixel_values, 75)
    print ("3Q      : ", third_quater)

    #標準偏差
    print ("SD      : " + str(np.std(_pixel_values)))

    print("\n")

    return mean, first_quater, median, third_quater

mean1, first_quater1, median1, third_quater1 = get_data_of_pixel_value(img_1_gray_nonzero)
mean2, first_quater2, median2, third_quater2 = get_data_of_pixel_value(img_2_gray_nonzero)




# ----------------------
# ----- Matplotlib -----
# ----------------------
ax[2].hist(img_1_gray_nonzero.ravel(), bins=255, color='r', alpha=0.5, label="Input image")
ax[2].hist(img_2_gray_nonzero.ravel(), bins=255, color='b', alpha=0.5, label="Input image (LR=1)")

# draw line
ax[2].axvline(mean1, color='black')
ax[2].axvline(median1, color='r')
ax[2].axvline(first_quater1, color='r')
ax[2].axvline(third_quater1, color='r')

ax[2].axvline(mean2, color='black')
ax[2].axvline(median2, color='b')
ax[2].axvline(first_quater2, color='b')
ax[2].axvline(third_quater2, color='b')

ax[2].set_title("Comparative histograms", fontsize=12)
ax[2].set_xlabel("Pixel value", fontsize=12)
ax[2].set_ylabel("Number of pixels", fontsize=12)
ax[2].set_xlim([-5, 260])

ax[2].legend(fontsize=12)

plt.show()