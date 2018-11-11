#
# リピートレベルごとの平均輝度値を出力するプログラム
# 背景の輝度値は平均計算から除外
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

img_in_RGB = read_img(args[1])
# image information（height × width × 色数）
print("img_in_RGB : ", img_in_RGB.shape)  
print("\n")



# ----------------------------
# ----- Show input image -----
# ----------------------------
def show_img(_i, _img, _img_name):
  ax[_i].set_title(_img_name)

  # show image
  ax[_i].imshow(_img)

  return

#show_img(0, img_in_RGB, "Input image")



# -------------------------------
# ----- Convert RGB to Gray -----
# -------------------------------
img_in_gray = cv2.cvtColor(img_in_RGB, cv2.COLOR_RGB2GRAY)
img_in_gray_nonzero = img_in_gray[img_in_gray>0]
print("img_in_gray : ", img_in_gray.shape)  
print("\n")



# -----------------------------------------------
# ----- Get statistical data of pixel value -----
# -----------------------------------------------
def get_data_of_pixel_value(_img, _img_name):
  print("===== Statistical Data of", _img_name, " =====")
  print("Num of pixel values (== 255) :", np.sum(_img == 255))
  #print("Num of pixel values (<= 1)   :", np.sum(_img <= 1))
  print("Num of pixel values (== 0)   :", np.sum(_img == 0) )
  print("\nMax :", np.max(_img))
  print("Min :", np.min(_img))
  # print("\nMean :", np.mean(_img))
  # print("SD  :", np.std(_img))
  print("Median :", np.median(_img))
  print("\nMean :", _img[_img != 0].mean())
  print("SD :", _img[_img != 0].std())

  N_all_nonzero = np.sum(_img > 0)
  print("\nN_all_nonzero\n>", N_all_nonzero, "(pixels)")
  print("\nratio\n>", round(np.sum(_img == 255)/N_all_nonzero, 4), "(%)")

  print("\n")
  
  return
  #return _img[_img != 0].mean()

get_data_of_pixel_value(img_in_gray_nonzero, args[1])
#get_data_of_pixel_value(img_in_RGB, args[1])
# mean   = get_data_of_pixel_value(gray_img_origin_RL1,   "img_original_RL1")



# ----------------------
# ----- Matplotlib -----
# ----------------------
ax[2].hist(img_in_gray_nonzero.ravel(), bins=50, color='red', alpha=0.5, label="Input image")
# ax[2].axvline(R_mean_noised_RL1, color='red')
# ax[2].axvline(R_mean_noised_RL100, color='blue')

ax[2].set_title("Histogram of Pixel Value of Gray Scale", fontsize=12)
ax[2].set_xlabel("Pixel value", fontsize=12)    # 画素値 
ax[2].set_ylabel("Number of pixels", fontsize=12) # 画素値の度数
ax[2].set_xlim([-5, 260])
#ax[2].set_ylim([0, 250000])
#plt.grid()
ax[2].legend(fontsize=12)

# fig.show()
#plt.show()

