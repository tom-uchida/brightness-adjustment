# ========================================
#       Convert Color Tone of Image
# ========================================
# BGR → RGB → HSV → Convert Color Tone → RGB

import cv2, matplotlib
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')



# -----------------------------
# ----- Placement setting -----
# -----------------------------
fig, ax = plt.subplots(3, figsize=(8, 6)) # figsize(width, height)
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

#img_in_RGB = read_img('images/noised_butai_600_340_RL10.jpg')
#img_in_RGB = read_img('images/noised_butai_600_340_RL100.jpg')
img_in_RGB = read_img('images/Lenna.jpg')
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
                " Input image(HSV)",
                int(img_in_HSV.shape[0]*0.5), 
                int(img_in_HSV.shape[0]*0.5))



# ------------------------------
# ----- Convert Color Tone -----
# ------------------------------
def convert_color_tone(_hsv_img, _val_param, _sat_param, _hue_param):
  # -----------------
  # ----- Value -----
  # -----------------
  #_hsv_img[:, :, 2] = np.add(_hsv_img[:, :, 2], _val_param)
  val = _hsv_img[:, :, 2]
  val = np.add(val, _val_param)

  # 0~255の範囲に収める
  val[val < 0]   = 0
  val[255 < val] = 255



  # ----------------------
  # ----- Saturation -----
  # ----------------------
  sat = _hsv_img[:, :, 1]
  sat = np.divide(sat, _sat_param)

  # 0~255の範囲に収める
  sat[sat < 0]   = 0
  sat[255 < sat] = 255



  # ---------------
  # ----- Hue -----
  # ---------------
  hue = _hsv_img[:, :, 0]

  # 0°~180°の範囲に収める
  hue[hue < 0]   = 0
  hue[179 < hue] = 179



  # Apply conversion
  img_HSV = np.empty((_hsv_img.shape[0], _hsv_img.shape[1], 3), dtype=np.uint8)
  img_HSV[:, :, 0] = hue
  img_HSV[:, :, 1] = sat
  img_HSV[:, :, 2] = val

  return img_HSV

# ===============================================
#      It is required to adjust parameters.
# ===============================================
img_out_HSV = convert_color_tone(img_in_HSV, 25, 2, 0)

show_HSV_values(img_out_HSV, 
                "Output image(HSV)", 
                int(img_in_HSV.shape[0]*0.5), 
                int(img_in_HSV.shape[0]*0.5))



# ------------------------------
# ----- Convert HSV to RGB -----
# ------------------------------
img_out_RGB = cv2.cvtColor(img_out_HSV, cv2.COLOR_HSV2RGB)

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
# Hue histogram
hist_img_in_hue,  bins_in_hue  = np.histogram(img_in_HSV[:,:,0].ravel(),  180, [0, 180])
hist_img_out_hue, bins_out_hue = np.histogram(img_out_HSV[:,:,0].ravel(), 180, [0, 180])

# plot
ax[2].set_title("Hue histogram")
# ax[2].hist(img_in_HSV[:,:,0].ravel(),  bins=50, alpha=0.4, label=" Input Image")
# ax[2].hist(img_out_HSV[:,:,0].ravel(), bins=50, alpha=0.4, label=" Output Image")
ax[2].plot(hist_img_in_hue,  label="Input Image")
ax[2].plot(hist_img_out_hue, label="Output Image")
ax[2].set_xlabel("Hue value")
ax[2].set_ylabel("Frequency")
ax[2].legend()
plt.show()



# ----------------------
# ----- Save image -----
# ----------------------
# cv2.imwrite('images/out.png', img_out_RGB)