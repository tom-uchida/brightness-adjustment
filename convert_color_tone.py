# Convert Color Tone of Image
# RGB → HSV → Convert Color Tone → RGB

import cv2, matplotlib
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')



# ------------------------------
# ----- Placement settings -----
# ------------------------------
fig, ax = plt.subplots(2, figsize=(10, 4)) # figsize(width, height)
ax[0] = plt.subplot2grid((1,2), (0,0))
ax[1] = plt.subplot2grid((1,2), (0,1))



# ----------------------------
# ----- Read input image -----
# ----------------------------
def read_img(_img_name):
  # read input image
  img_BGR = cv2.imread(_img_name)

  # convert color (BGR → RGB)
  img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

  return img_RGB

img_in_RGB = read_img('images/noised_butai_600_340_RL100.jpg')
print("Input image(RGB) : ", img_in_RGB.shape) # （height × width × 色数）
print("\n")



# ------------------------------
# ----- Convert RGB to HSV -----
# ------------------------------
# Hue[0,179], Saturation[0,255], Value[0,255]
img_in_HSV = cv2.cvtColor(img_in_RGB, cv2.COLOR_RGB2HSV)

def show_HSV_values(_hsv_img, _hsv_img_name, _y, _x):
  print(_hsv_img_name,"[",_y,",",_x,"] → (H S V) = (",
        _hsv_img[_y, _x, 0], _hsv_img[_y, _x, 1], _hsv_img[_y, _x, 2],")")
  return

show_HSV_values(img_in_HSV, 
                "Input image(HSV)", 
                int(img_in_HSV.shape[0]*0.5), 
                int(img_in_HSV.shape[0]*0.5))



# ------------------------------
# ----- Convert Color Tone -----
# ------------------------------
def convert_color_tone(_hsv_img, _hue, _sat, _val):
  # -----------------
  # ----- Value -----
  # -----------------
  #_hsv_img[:, :, 2] = np.add(_hsv_img[:, :, 2], _val)
  hsv_img_val = _hsv_img[:, :, 2]
  hsv_img_val = np.add(hsv_img_val, _val)

  # 0~255の範囲に収める
  hsv_img_val[hsv_img_val < 0]   = 0
  hsv_img_val[255 < hsv_img_val] = 255

  # apply
  _hsv_img[:, :, 2] = hsv_img_val



  # ----------------------
  # ----- Saturation -----
  # ----------------------
  hsv_img_sat = _hsv_img[:, :, 1]
  hsv_img_sat = np.divide(hsv_img_sat, _sat)

  # 0~255の範囲に収める
  hsv_img_sat[hsv_img_sat < 0]   = 0
  hsv_img_sat[255 < hsv_img_sat] = 255

  # apply
  _hsv_img[:, :, 1] = hsv_img_sat


  # ---------------
  # ----- Hue -----
  # ---------------
  hsv_img_hue = _hsv_img[:, :, 0]
  #hsv_img_hue = np.subtract(hsv_img_hue, _hue)

  # 0~255の範囲に収める
  hsv_img_hue[hsv_img_sat < 0]   = 0
  hsv_img_hue[170 < hsv_img_sat] = 179

  # apply
  _hsv_img[:, :, 0] = hsv_img_hue



  return

# ===============================================
#      It is required to adjust parameters.
# ===============================================
convert_color_tone(img_in_HSV, 0, 1.5, 10)

show_HSV_values(img_in_HSV, 
                "Output image(HSV)", 
                int(img_in_HSV.shape[0]*0.5), 
                int(img_in_HSV.shape[0]*0.5))



# ------------------------------
# ----- Convert HSV to RGB -----
# ------------------------------
img_out_RGB = cv2.cvtColor(img_in_HSV, cv2.COLOR_HSV2RGB)

def show_RGB_values(_rgb_img, _rgb_img_name, _y, _x):
  print(_rgb_img_name,"[",_y,",",_x,"] → (R G B) = (",
        _rgb_img[_y, _x, 0], _rgb_img[_y, _x, 1], _rgb_img[_y, _x, 2],")")
  return

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



# ----------------------------------------------------
# ----- Compare Histograms of Input/Output image -----
# ----------------------------------------------------



# ----------------------
# ----- Save image -----
# ----------------------
# cv2.imwrite('images/out.png', img_out_RGB)



plt.show()