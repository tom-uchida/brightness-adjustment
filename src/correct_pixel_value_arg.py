import cv2, matplotlib
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import sys
args = sys.argv

plt.style.use('seaborn-white')

from matplotlib import cycler
colors = cycler('color', ['#EE6666', '#3388BB', '#9988DD', '#EECC55', '#88BB44', '#FFBBBB'])
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True, grid=False, prop_cycle=colors)
plt.rc('grid', color='w', linestyle='solid')
plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=2)



# -----------------------------
# ----- Placement setting -----
# -----------------------------
fig, ax = plt.subplots(3, figsize=(8, 8)) # figsize(width, height)
ax[0] = plt.subplot2grid((2,2), (0,0))
ax[1] = plt.subplot2grid((2,2), (0,1))
ax[2] = plt.subplot2grid((2,2), (1,0), colspan=2)


# ----------------------------
# ----- Read input image -----
# ----------------------------
def read_img(_img_name):
  # read input image
  img_BGR = cv2.imread(_img_name)

  # convert color (BGR → RGB)
  img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

  return img_RGB



# -----------------------
# ----- Pre-Process -----
# -----------------------
# Read input image
img_in_RGB = read_img(args[1])

# correct parameter
param = float(args[2])

# print("Input image(RGB) : ", img_in_RGB.shape) # （height × width × 色数）
# print('R Max:',np.max(img_in_RGB[:, :, 0]),' Min:',np.min(img_in_RGB[:, :, 0]))
# print('G Max:',np.max(img_in_RGB[:, :, 1]),' Min:',np.min(img_in_RGB[:, :, 1]))
# print('B Max:',np.max(img_in_RGB[:, :, 2]),' Min:',np.min(img_in_RGB[:, :, 2]))
# print("\n")



# -------------------------------
# ----- Correct pixel value -----
# -------------------------------
def correct_pixel_value(_rgb_img, _param):
  corrected_img_RGB = np.empty((_rgb_img.shape[0], _rgb_img.shape[1], 3), dtype=np.uint8)
  corrected_img_RGB[:, :, 0] = cv2.multiply(_rgb_img[:, :, 0], _param)
  corrected_img_RGB[:, :, 1] = cv2.multiply(_rgb_img[:, :, 1], _param)
  corrected_img_RGB[:, :, 2] = cv2.multiply(_rgb_img[:, :, 2], _param)

  return corrected_img_RGB

img_out_RGB = correct_pixel_value(img_in_RGB, param)
# print('R Max:',np.max(img_out_RGB[:, :, 0]),' Min:',np.min(img_out_RGB[:, :, 0]))
# print('G Max:',np.max(img_out_RGB[:, :, 1]),' Min:',np.min(img_out_RGB[:, :, 1]))
# print('B Max:',np.max(img_out_RGB[:, :, 2]),' Min:',np.min(img_out_RGB[:, :, 2]))
# print("\n")



# -------------------------------
# ----- Convert RGB to Gray -----
# -------------------------------
img_in_gray = cv2.cvtColor(img_in_RGB, cv2.COLOR_RGB2GRAY)
img_out_gray = cv2.cvtColor(img_out_RGB, cv2.COLOR_RGB2GRAY)

# exclude pixel value == 0
img_in_gray_nonzero = img_in_gray[img_in_gray > 0]
img_out_gray_nonzero = img_out_gray[img_out_gray > 0]



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

get_data_of_pixel_value(img_in_gray_nonzero)
get_data_of_pixel_value(img_out_gray_nonzero)



# -----------------------
# ----- Show images -----
# -----------------------
def show_img(_i, _img, _img_name):
  # image title
  ax[_i].set_title(_img_name)

  # show image
  ax[_i].imshow(_img)

  return

show_img(0, img_in_RGB,  "Input Image")
show_img(1, img_out_RGB, "Corrected Image")



# -------------------------------
# ----- Convert RGB to Gray -----
# -------------------------------
img_in_gray  = cv2.cvtColor(img_in_RGB,  cv2.COLOR_RGB2GRAY)
img_out_gray = cv2.cvtColor(img_out_RGB, cv2.COLOR_RGB2GRAY)
img_in_gray_nonzero = img_in_gray[img_in_gray>0]
img_out_gray_nonzero = img_out_gray[img_out_gray>0]



# --------------------------------------------------------
# ----- Compare two histograms of Input/Output image -----
# --------------------------------------------------------
# plot
ax[2].set_title("Comparative Histogram")
ax[2].hist(img_in_gray_nonzero.ravel(),  bins=50, color='red',  alpha=0.4, label=" Input Image")
ax[2].hist(img_out_gray_nonzero.ravel(), bins=50, color='blue', alpha=0.4, label=" Corrected Image")
ax[2].set_xlabel("Pixel value")
ax[2].set_ylabel("Number of pixels")
ax[2].legend()
ax[2].set_xlim([-5, 260])
#ax[2].set_ylim([0, 15000])
plt.show()



# ----------------------
# ----- Save image -----
# ----------------------
img_out_BGR = cv2.cvtColor(img_out_RGB, cv2.COLOR_RGB2BGR)
cv2.imwrite('images/out.bmp', img_out_BGR)