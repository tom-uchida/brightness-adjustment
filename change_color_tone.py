# ======================================
#       Change Color Tone of Image
# ======================================
# BGR → RGB → Change Color Tone

import cv2, matplotlib
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')



# -----------------------------
# ----- Placement setting -----
# -----------------------------
fig, ax = plt.subplots(3, figsize=(10, 8)) # figsize(width, height)
ax[0] = plt.subplot2grid((2,2), (0,0))
ax[1] = plt.subplot2grid((2,2), (0,1))
ax[2] = plt.subplot2grid((2,2), (1,0), colspan=2)



# -------------------------------
# ----- Select input image  -----
# -------------------------------
#image_name = 'images/noised_butai_600_340_RL100.jpg'
#image_name = 'images/noised_butai_600_340_RL10.jpg'
#image_name = 'images/out_gaussian_RL100.jpg'
#image_name = 'images/out_poisson_RL100.jpg'
image_name = 'images/out_spike_RL100.jpg'
r_param, g_param, b_param = 2,2,2



# ----------------------------
# ----- Read input image -----
# ----------------------------
def read_img(_img_name):
  # read input image
  img_BGR = cv2.imread(_img_name)

  # convert color (BGR → RGB)
  img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

  return img_RGB

img_in_RGB = read_img(image_name)
print("Input image(RGB) : ", img_in_RGB.shape) # （height × width × 色数）



# ------------------------------
# ----- Change color tone -----
# ------------------------------
def change_color_tone(_rgb_img, _r_param, _g_param, _b_param):
  # Red
  red = cv2.multiply(_rgb_img[:, :, 0], _r_param)

  # Green
  green = cv2.multiply(_rgb_img[:, :, 1], _g_param)

  # Blue
  blue = cv2.multiply(_rgb_img[:, :, 2], _b_param)

  # Apply change
  revised_img_RGB = np.empty((_rgb_img.shape[0], _rgb_img.shape[1], 3), dtype=np.uint8)
  revised_img_RGB[:, :, 0] = red
  revised_img_RGB[:, :, 1] = green
  revised_img_RGB[:, :, 2] = blue

  return revised_img_RGB

# ===============================================
#      It is required to adjust parameters.
# ===============================================
img_out_RGB = change_color_tone(img_in_RGB, r_param, g_param, b_param)
print('R Max:',np.max(img_out_RGB[:, :, 0]),' Min:',np.min(img_out_RGB[:, :, 0]))
print('G Max:',np.max(img_out_RGB[:, :, 1]),' Min:',np.min(img_out_RGB[:, :, 1]))
print('B Max:',np.max(img_out_RGB[:, :, 2]),' Min:',np.min(img_out_RGB[:, :, 2]))
print("\n")



# def show_RGB_values(_rgb_img, _rgb_img_name, _y, _x):
#   print(_rgb_img_name,"[",_y,",",_x,"] → (R G B) = (",
#         _rgb_img[_y, _x, 0], _rgb_img[_y, _x, 1], _rgb_img[_y, _x, 2],")")
#   return

# show_RGB_values(img_in_RGB, 
#                 "Input  image(RGB)", 
#                 int(img_in_RGB.shape[0]*0.5), 
#                 int(img_in_RGB.shape[0]*0.5))

# show_RGB_values(img_out_RGB, 
#                 "Output image(RGB)", 
#                 int(img_out_RGB.shape[0]*0.5), 
#                 int(img_out_RGB.shape[0]*0.5))



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
show_img(1, img_out_RGB, "Output Image")



# -------------------------------
# ----- Convert RGB to Gray -----
# -------------------------------
img_in_gray  = cv2.cvtColor(img_in_RGB,  cv2.COLOR_RGB2GRAY)
img_out_gray = cv2.cvtColor(img_out_RGB, cv2.COLOR_RGB2GRAY)



# --------------------------------------------------------
# ----- Compare two histograms of Input/Output image -----
# --------------------------------------------------------
# plot
ax[2].set_title("Comparison of Two Histogram")
ax[2].hist(img_in_gray.ravel(),  bins=50, color='red',  alpha=0.4, label=" Input Image")
ax[2].hist(img_out_gray.ravel(), bins=50, color='blue', alpha=0.4, label=" Output Image")
ax[2].set_xlabel("Pixel value", fontsize=12)
ax[2].set_ylabel("Number of pixels", fontsize=12)
ax[2].legend(fontsize=16)
ax[2].set_ylim([0, 50000])
plt.show()



# ----------------------
# ----- Save image -----
# ----------------------
img_out_BGR = cv2.cvtColor(img_out_RGB, cv2.COLOR_RGB2BGR)
cv2.imwrite('images/out.jpg', img_out_BGR)