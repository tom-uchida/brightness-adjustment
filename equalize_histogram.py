#
# ヒストグラムを平坦化するプログラム
# 1chずつしか平坦化できないので注意
#

import cv2, matplotlib
import numpy as np
import matplotlib.pyplot as plt
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
fig, ax = plt.subplots(4, figsize=(9, 8)) # figsize(width, height)
fig.subplots_adjust(hspace=0.4, wspace=0.5) # interval
ax[0] = plt.subplot2grid((2,2), (0,0))
ax[1] = plt.subplot2grid((2,2), (0,1))
ax[2] = plt.subplot2grid((2,2), (1,0))
ax[3] = plt.subplot2grid((2,2), (1,1))



# ----------------------------
# ----- Read input image -----
# ----------------------------
def read_img(_img_name):
    # read input image
    img_BGR = cv2.imread(_img_name)

    # convert color (BGR → RGB)
    img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

    return img_RGB

# read input image
img_in_RGB = read_img("images/2018-09-30/out_gaussian_RL100.jpg")
#img_in_RGB = read_img("images/2018-10-01/out-All.bmp")

# convert RGB to Gray
#img_in_gray = cv2.cvtColor(img_in_RGB, cv2.COLOR_RGB2GRAY)

# convert RGB to HSV
# Hue[0,179], Saturation[0,255], Value[0,255]
#img_in_HSV = cv2.cvtColor(img_in_RGB, cv2.COLOR_RGB2HSV)



# ------------------------------
# ----- Equalize histogram -----
# ------------------------------
def equalize_hist(_img_in_1ch):
    # get height and width of input image
    h, w = _img_in_1ch.shape[0], _img_in_1ch.shape[1]
    
    # calc size
    S = w*h
    
    # get max pixel value
    #I_max = _img_in_1ch.max()
    I_max = _img_in_1ch.max()-50
    
    # make histogram
    hist, bins = np.histogram(_img_in_1ch.ravel(), 256, [0,256])
    # print(S)
    # print(np.sum(hist[:255]))

    # make empty numpy array for output image 
    img_out_1ch = np.empty( (h,w) )

    # equalize histogram
    for y in range(0, h):
        for x in range(0, w):
            img_out_1ch[y][x] = I_max * (np.sum( hist[: _img_in_1ch[y][x]]) / S)
            #img_out_1ch[y][x] = img_out_1ch[y][x] - 50

    return img_out_1ch

# equalize histogram
#img_out_gray = equalize_hist(img_in_gray)
img_out_RGB = img_in_RGB.copy()
img_out_RGB[:,:,0] = equalize_hist(img_in_RGB[:,:,0])
img_out_RGB[:,:,1] = equalize_hist(img_in_RGB[:,:,1])
img_out_RGB[:,:,2] = equalize_hist(img_in_RGB[:,:,2])

# convert HSV to RGB
#img_in_HSV[:,:,2] = equalize_hist(img_in_HSV[:,:,2]) # Value
#img_out_RGB = cv2.cvtColor(img_in_HSV, cv2.COLOR_HSV2RGB)



# -----------------------
# ----- Show images -----
# -----------------------
def show_img(_i, _img, _img_name):
  ax[_i].set_title(_img_name)

  # show image
  ax[_i].imshow(_img)
  #ax[_i].imshow(_img, cmap='gray')

  return

show_img(0, img_in_RGB,   "Original image")
show_img(1, img_out_RGB,  "Improved image")



# -----------------------------------------------
# ----- Get statistical data of pixel value -----
# -----------------------------------------------
def get_data_of_pixel_value(_pixel_values):
  print("===== Statistical Data of Pixel Values =====")
  print("> Max    : ", np.max(_pixel_values))
  print("> Min    : ", np.min(_pixel_values))
  print("> Mean   : ", np.mean(_pixel_values))
  print("> S.D    : ", np.std(_pixel_values))
  print("(nonzero)")
  print("\n")
  return 

# img_in_RGB_nonzero = img_in_RGB[img_in_RGB > 0]
# img_out_RGB_nonzero = img_out_RGB[img_out_RGB > 0]
img_in_R_nonzero = img_in_RGB[:,:,0][img_in_RGB[:,:,0] > 0]
img_in_G_nonzero = img_in_RGB[:,:,1][img_in_RGB[:,:,1] > 0]
img_in_B_nonzero = img_in_RGB[:,:,2][img_in_RGB[:,:,2] > 0]
img_out_R_nonzero = img_out_RGB[:,:,0][img_out_RGB[:,:,0] > 0]
img_out_G_nonzero = img_out_RGB[:,:,1][img_out_RGB[:,:,1] > 0]
img_out_B_nonzero = img_out_RGB[:,:,2][img_out_RGB[:,:,2] > 0]
# get_data_of_pixel_value(img_in_gray_nonzero)
# get_data_of_pixel_value(img_out_gray_nonzero)
# get_data_of_pixel_value(img_in_RGB_nonzero)
# get_data_of_pixel_value(img_out_RGB_nonzero)



# -----------------------
# ----- Save images -----
# -----------------------
img_in_BGR = cv2.cvtColor(img_in_RGB, cv2.COLOR_RGB2BGR)
img_out_BGR = cv2.cvtColor(img_out_RGB, cv2.COLOR_RGB2BGR)
cv2.imwrite("images/output_before.jpg", img_in_BGR)
cv2.imwrite("images/output_after.jpg",  img_out_BGR)



# ----------------------
# ----- Matplotlib -----
# ----------------------
# original image
ax[2].hist(img_in_R_nonzero.ravel(), bins=50, color='r', alpha=0.5, label="R")
ax[2].hist(img_in_G_nonzero.ravel(), bins=50, color='g', alpha=0.5, label="G")
ax[2].hist(img_in_B_nonzero.ravel(), bins=50, color='b', alpha=0.5, label="B")

# improved image
ax[3].hist(img_out_R_nonzero.ravel(), bins=50, color='r', alpha=0.5, label="R")
ax[3].hist(img_out_G_nonzero.ravel(), bins=50, color='g', alpha=0.5, label="G")
ax[3].hist(img_out_B_nonzero.ravel(), bins=50, color='b', alpha=0.5, label="B")

# ----- Text setting -----
#props = dict(boxstyle='round', facecolor='red', alpha=0.5)
# ax[2].text(0.82, 0.88, "mean:255", transform=ax[2].transAxes, fontsize=14, color='r')
#ax[2].text(0.2, 0.25, "mean:41", transform=ax[2].transAxes, fontsize=14, color='r')
#ax[2].text(0.55, 0.4, "mean:164", transform=ax[2].transAxes, fontsize=14, color='r')
#ax[2].text(0.05, 0.25, "mean:39", transform=ax[2].transAxes, fontsize=14, color='b')

ax[2].set_title("Histogram")
ax[2].set_xlabel("Pixel value")
ax[2].set_ylabel("Number of pixels")
ax[2].set_xlim([-10, 266])
#ax[2].set_ylim([0, 40000])
ax[2].legend()

ax[3].set_title("Histogram")
ax[3].set_xlabel("Pixel value")
ax[3].set_ylabel("Number of pixels")
ax[3].set_xlim([-10, 266])
#ax[3].set_ylim([0, 40000])
ax[3].legend()

fig.show()
plt.show()

