import cv2, matplotlib
import matplotlib.pyplot as plt
import numpy as np
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
fig, ax = plt.subplots(3, figsize=(8, 6)) # figsize(width, height)
fig.subplots_adjust(hspace=0.4, wspace=0.4) # interval
ax[0] = plt.subplot2grid((2,2), (0,0))
ax[1] = plt.subplot2grid((2,2), (0,1))
ax[2] = plt.subplot2grid((2,2), (1,0), colspan=2)



# ----------------------------
# ----- Read input image -----
# ----------------------------
def read_img_with_opacity(_img_name):
	# read input image
	img = cv2.imread(_img_name, -1)

	# convert color (BGR → RGB)
	img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

	return img



img_1 = read_img_with_opacity(args[1])
img_2 = read_img_with_opacity(args[2])
# image information（height × width × channel）
# print("\nimg_in : ", img1.shape)



# ----------------------------
# ----- Show input image -----
# ----------------------------
def show_img(_i, _img, _img_name):
    ax[_i].set_title(_img_name)

    # show image
    ax[_i].imshow(_img)

    return

show_img(0, img_1,  "Original image")
show_img(1, img_2,  "Corrected image")



# -------------------------------------------
# ----- Get statistical data of opacity -----
# -------------------------------------------
def get_data_of_pixel_value(_opacities):
    print("===== Statistical Data of Opacity =====")
    print("> Max    : ", np.max(_opacities))
    print("> Min    : ", np.min(_opacities))
    print("> Mean   : ", np.mean(_opacities))
    print("\n")

    return 

get_data_of_pixel_value(img_1[:, :, 3])
get_data_of_pixel_value(img_2[:, :, 3])



# -----------------------------
# ----- Opacity Histogram -----
# -----------------------------
alpha_1 = img_1[:, :, 3]
alpha_2 = img_2[:, :, 3]
ax[2].hist(alpha_1.ravel(), bins=50, color='r', alpha=0.5, label="Original image")
ax[2].hist(alpha_2.ravel(), bins=50, color='b', alpha=0.5, label="Corrected image")

ax[2].set_title("Comparative histogram")
ax[2].set_xlabel("Alpha value")
ax[2].set_ylabel("Number")
ax[2].set_xlim([-5, 260])
#ax[2].set_ylim([0, 750000])
ax[2].legend()

plt.show()
