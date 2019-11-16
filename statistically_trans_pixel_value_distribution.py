import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cycler
import matplotlib.gridspec as gridspec
import matplotlib.patches as pat
import cv2
import subprocess
import sys
import statistics
import time

# Graph settings
plt.style.use('seaborn-white')
colors = cycler('color', ['#EE6666', '#3388BB', '#9988DD', '#EECC55', '#88BB44', '#FFBBBB'])
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True, grid=False, prop_cycle=colors)
plt.rc('grid', color='w', linestyle='solid')
plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=2)
# plt.rcParams['font.family'] = 'IPAGothic' # font setting
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["mathtext.rm"] = "Times New Roman"


# Check arguments
args = sys.argv
if len(args) != 2:
    print("\n")
    print("USAGE   : $ python *.py [input_image_data]")
    print("EXAMPLE : $ python *.py [input_image.bmp]")
    #raise Exception
    sys.exit()


BGColor              = [0, 0, 0] # Background color
BGColor_Gray         = np.uint8(0.299*BGColor[0]+0.587*BGColor[1]+0.114*BGColor[2])


# Read Input Image
def readImage(_img_name):
    # read input image
    img_BGR = cv2.imread(_img_name)

    # convert color BGR to RGB
    img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

    return img_RGB



# RGB Histogram
def rgbHist(_img_rgb, _ax, _title):
    R_nonzero = _img_rgb[:,:,0][_img_rgb[:,:,0] != BGColor[0]]
    G_nonzero = _img_rgb[:,:,1][_img_rgb[:,:,1] != BGColor[1]]
    B_nonzero = _img_rgb[:,:,2][_img_rgb[:,:,2] != BGColor[2]]
    _ax.hist(R_nonzero.ravel(), bins=bin_number, color='r', alpha=0.5, label="R")
    _ax.hist(G_nonzero.ravel(), bins=bin_number, color='g', alpha=0.5, label="G")
    _ax.hist(B_nonzero.ravel(), bins=bin_number, color='b', alpha=0.5, label="B")
    _ax.legend()

    _ax.set_title(_title)
    _ax.set_xlim([-5, 260])
    
    return _ax



# Grayscale Histogram
def grayscaleHist(_img_gray, _ax, _title):
    img_Gray_nonzero = _img_gray[_img_gray != BGColor_Gray]
    _ax.hist(img_Gray_nonzero.ravel(), bins=bin_number, color='black', alpha=1.0)

    _ax.set_title(_title)
    _ax.set_xlim([-5, 260])
    
    return _ax



# Histograms of Input image(L=1), Input image and Adjusted image
def comparativeHist(_img_in_rgb_L1, _img_in_rgb, _img_out_rgb, _ax, _y_max):
    # Convert RGB to Grayscale
    img_in_Gray_L1             = cv2.cvtColor(_img_in_rgb_L1, cv2.COLOR_RGB2GRAY)
    img_in_Gray_L1_non_bgcolor = img_in_Gray_L1[img_in_Gray_L1 != BGColor_Gray]
    img_in_Gray                 = cv2.cvtColor(_img_in_rgb, cv2.COLOR_RGB2GRAY)
    img_in_Gray_non_bgcolor     = img_in_Gray[img_in_Gray != BGColor_Gray]
    img_out_Gray                = cv2.cvtColor(_img_out_rgb, cv2.COLOR_RGB2GRAY)
    img_out_Gray_non_bgcolor    = img_out_Gray[img_out_Gray != BGColor_Gray]
    
    # input image(L=1)
    mean_in_L1 = int(np.mean(img_in_Gray_L1_non_bgcolor))
    _ax.hist(img_in_Gray_L1_non_bgcolor.ravel(), bins=bin_number, alpha=0.5, label="Input image ($L_{\mathrm{R}}=1$)", color='#1F77B4')
    _ax.axvline(mean_in_L1, color='#1F77B4')
    _ax.text(mean_in_L1+5, _y_max*0.8, "mean:"+str(mean_in_L1), color='#1F77B4', fontsize='12')

    # input image
    mean_in = int(np.mean(img_in_Gray_non_bgcolor))
    _ax.hist(img_in_Gray_non_bgcolor.ravel(), bins=bin_number, alpha=0.5, label="Input image", color='#FF7E0F')
    _ax.axvline(mean_in, color='#FF7E0F')
    _ax.text(mean_in+5, _y_max*0.6, "mean:"+str(mean_in), color='#FF7E0F', fontsize='12')

    # adjusted image
    mean_out = int(np.mean(img_out_Gray_non_bgcolor))
    _ax.hist(img_out_Gray_non_bgcolor.ravel(), bins=bin_number, alpha=0.5, label="Adjusted image", color='#2C9F2C')
    _ax.axvline(mean_out, color='#2C9F2C')
    _ax.text(mean_out+5, _y_max*0.7, "mean:"+str(mean_out), color='#2C9F2C', fontsize='12')

    _ax.set_title('Comparative histogram')
    _ax.set_xlabel("Pixel value")
    _ax.set_ylabel("Number of pixels")
    _ax.legend(fontsize='12')
    
    return _ax



# Decompose the input image
def separateBackgroundColor():
    num_of_bgcolor      = np.count_nonzero(b_index_bgcolor)
    num_of_non_bgcolor  = np.count_nonzero(b_index_non_bgcolor)
    print("Num of Background Color           :", num_of_bgcolor)
    print("Num of Non-Background Color       :", num_of_non_bgcolor)
    print("The ratio of Background Color     :", round(num_of_bgcolor/N_all*100), "(%)")

    # Apply decomposition
    bg_R     = np.where(b_index_bgcolor,     BGColor[0],        0)
    bg_G     = np.where(b_index_bgcolor,     BGColor[1],        0)
    bg_B     = np.where(b_index_bgcolor,     BGColor[2],        0)
    non_bg_R = np.where(b_index_non_bgcolor, img_in_RGB[:,:,0], 0)
    non_bg_G = np.where(b_index_non_bgcolor, img_in_RGB[:,:,1], 0)
    non_bg_B = np.where(b_index_non_bgcolor, img_in_RGB[:,:,2], 0)

    # Create BGColor image and Non-BGColor image
    img_in_RGB_bgcolor, img_in_RGB_non_bgcolor = img_in_RGB.copy(), img_in_RGB.copy()
    img_in_RGB_bgcolor[:,:,0], img_in_RGB_bgcolor[:,:,1], img_in_RGB_bgcolor[:,:,2] = bg_R, bg_G, bg_B
    img_in_RGB_non_bgcolor[:,:,0], img_in_RGB_non_bgcolor[:,:,1], img_in_RGB_non_bgcolor[:,:,2] = non_bg_R,non_bg_G, non_bg_B

    return img_in_RGB_bgcolor, img_in_RGB_non_bgcolor



def transformPixelValueDistributionStatistically():
    # img = (img - np.mean(img))/np.std(img)*16+64
    tmp_img_uint8 = img_in_RGB_non_bgcolor.copy()
    tmp_img_float = tmp_img_uint8.astype(float)
    
    # Make the mean pixel value "0"
    tmp_img_float[:,:,0] = cv2.subtract(tmp_img_float[:,:,0],   float(mean_pixel_value)) # R
    tmp_img_float[:,:,1] = cv2.subtract(tmp_img_float[:,:,1],   float(mean_pixel_value)) # G
    tmp_img_float[:,:,2] = cv2.subtract(tmp_img_float[:,:,2],   float(mean_pixel_value)) # B

    # Make the std pixel value "ideal_std_pixel_value"
    multiply_value = ideal_std_pixel_value / std_pixel_value
    tmp_img_float[:,:,0] = cv2.multiply(tmp_img_float[:,:,0],   float(multiply_value))
    tmp_img_float[:,:,1] = cv2.multiply(tmp_img_float[:,:,1],   float(multiply_value))
    tmp_img_float[:,:,2] = cv2.multiply(tmp_img_float[:,:,2],   float(multiply_value))

    # Make the mean pixel value "ideal_mean_pixel_value"
    tmp_img_float[:,:,0] = cv2.add(tmp_img_float[:,:,0],        float(ideal_mean_pixel_value))
    tmp_img_float[:,:,1] = cv2.add(tmp_img_float[:,:,1],        float(ideal_mean_pixel_value))
    tmp_img_float[:,:,2] = cv2.add(tmp_img_float[:,:,2],        float(ideal_mean_pixel_value))

    # 原点に揃え
    tmp_img_float = tmp_img_float - np.min(tmp_img_float)

    # 強引な調整
    tmp_R_non_bgcolor = tmp_img_float[:,:,0][b_index_non_bgcolor]
    tmp_G_non_bgcolor = tmp_img_float[:,:,1][b_index_non_bgcolor]
    tmp_B_non_bgcolor = tmp_img_float[:,:,2][b_index_non_bgcolor]
    tmp_R_mean, tmp_G_mean, tmp_B_mean = np.mean(tmp_R_non_bgcolor), np.mean(tmp_G_non_bgcolor), np.mean(tmp_B_non_bgcolor)
    tmp_R_std, tmp_G_std, tmp_B_std = np.std(tmp_R_non_bgcolor), np.std(tmp_G_non_bgcolor), np.std(tmp_B_non_bgcolor)
    b_R_outlier = tmp_img_float[:,:,0] >= tmp_R_mean + 1*tmp_R_std
    b_G_outlier = tmp_img_float[:,:,1] >= tmp_G_mean + 1*tmp_G_std
    b_B_outlier = tmp_img_float[:,:,2] >= tmp_B_mean + 1*tmp_B_std
    tmp_img_float[:,:,0] = np.where(b_R_outlier, tmp_img_float[:,:,0]-tmp_R_mean*0.5, tmp_img_float[:,:,0])
    tmp_img_float[:,:,1] = np.where(b_G_outlier, tmp_img_float[:,:,1]-tmp_G_mean*0.5, tmp_img_float[:,:,1])
    tmp_img_float[:,:,2] = np.where(b_B_outlier, tmp_img_float[:,:,2]-tmp_B_mean*0.5, tmp_img_float[:,:,2])

    # Convert float to np.uint8
    tmp_img_uint8 = tmp_img_float.astype(np.uint8)

    # Exclude background color from calculation
    pre_processed_img_in_RGB = tmp_img_uint8
    pre_processed_img_in_RGB[:,:,0] = np.where(b_index_non_bgcolor, tmp_img_uint8[:,:,0], 0) # R
    pre_processed_img_in_RGB[:,:,1] = np.where(b_index_non_bgcolor, tmp_img_uint8[:,:,1], 0) # G
    pre_processed_img_in_RGB[:,:,2] = np.where(b_index_non_bgcolor, tmp_img_uint8[:,:,2], 0) # B
    print("\nStatistically, transformed pixel value distribution.")

    # Save image
    pre_processed_img_in_BGR = cv2.cvtColor(pre_processed_img_in_RGB, cv2.COLOR_RGB2BGR)
    cv2.imwrite("images/transformed.bmp", pre_processed_img_in_BGR)

    # Create figure
    fig = plt.figure(figsize=(8, 6)) # figsize=(width, height)
    gs  = gridspec.GridSpec(2,2)

    ax1 = fig.add_subplot(gs[0,0])
    ax1.set_title('Before')
    ax1.imshow(img_in_RGB)
    ax1.set_xticks([]), ax1.set_yticks([])

    ax2 = fig.add_subplot(gs[0,1])
    ax2.set_title('After')
    ax2.imshow(pre_processed_img_in_RGB)
    ax2.set_xticks([]), ax2.set_yticks([])

    ax3 = fig.add_subplot(gs[1,0])
    ax3 = rgbHist(img_in_RGB, ax3, "Before")
    ax3.axvline(ideal_mean_pixel_value, color='red')

    ax4 = fig.add_subplot(gs[1,1])
    ax4 = rgbHist(pre_processed_img_in_RGB, ax4, "After")
    ax4.axvline(np.mean(tmp_img_float), color='red')
    ax4.axvline(tmp_R_mean + 1*tmp_R_std, color='green')
    ax4.axvline(tmp_R_mean + 2*tmp_R_std, color='blue')
    ax4.axvline(tmp_R_mean + 3*tmp_R_std, color='yellow')
    # ax4.set_ylim([0, 60000])

    plt.show()

    return pre_processed_img_in_RGB



def mappingPixelValue():
    img_in_Gray_non_bgcolor = cv2.cvtColor(img_in_RGB_non_bgcolor, cv2.COLOR_RGB2GRAY)
    min_pixel_value = img_in_Gray_non_bgcolor.min()
    max_pixel_value = img_in_Gray_non_bgcolor.max()

    # Mapping
    tmp_img_uint8 = img_in_RGB.copy()
    tmp_img_float = tmp_img_uint8.astype(float)
    
    tmp_img_float[:,:,0] = cv2.subtract(tmp_img_uint8[:,:,0],   float(min_pixel_value)) # R
    tmp_img_float[:,:,1] = cv2.subtract(tmp_img_uint8[:,:,1],   float(min_pixel_value)) # G
    tmp_img_float[:,:,2] = cv2.subtract(tmp_img_uint8[:,:,2],   float(min_pixel_value)) # B

    tmp_img_float[:,:,0] = cv2.divide(tmp_img_float[:,:,0],     float(max_pixel_value-min_pixel_value))
    tmp_img_float[:,:,1] = cv2.divide(tmp_img_float[:,:,1],     float(max_pixel_value-min_pixel_value))
    tmp_img_float[:,:,2] = cv2.divide(tmp_img_float[:,:,2],     float(max_pixel_value-min_pixel_value))

    tmp_img_float[:,:,0] = cv2.multiply(tmp_img_float[:,:,0],   float(threshold_pixel_value-min_pixel_value))
    tmp_img_float[:,:,1] = cv2.multiply(tmp_img_float[:,:,1],   float(threshold_pixel_value-min_pixel_value))
    tmp_img_float[:,:,2] = cv2.multiply(tmp_img_float[:,:,2],   float(threshold_pixel_value-min_pixel_value))

    tmp_img_uint8 = tmp_img_float.astype(np.uint8)

    mapped_img_in_RGB = tmp_img_uint8
    print("Mapping done.")

    return mapped_img_in_RGB



def preProcess(_img_RGB):
    print("Input image (RGB)                 :", _img_RGB.shape) # (height, width, channel)

    # Calc all number of pixels of the input image
    N_all = _img_RGB.shape[0] * _img_RGB.shape[1]
    print("N_all                             :", N_all, "(pixels)")

    # Exclude background color
    img_in_Gray_non_bgcolor = img_in_Gray[img_in_Gray != BGColor_Gray]
    
    # Calc the number of pixels excluding background color
    N_all_non_bgcolor       = np.sum(img_in_Gray != BGColor_Gray)
    print("N_all_non_bgcolor                 :", N_all_non_bgcolor, "(pixels)")

    # Calc mean pixel value
    max_pixel_value         = np.max(img_in_Gray_non_bgcolor)
    print("Max pixel value                   :", max_pixel_value, "(pixel value)")

    # Calc mean pixel value
    mean_pixel_value        = np.mean(img_in_Gray_non_bgcolor)
    print("Mean pixel value                  :", round(mean_pixel_value, 1), "(pixel value)")

    # Calc std pixel value
    std_pixel_value         = np.std(img_in_Gray_non_bgcolor)
    print("Std pixel value                   :", round(std_pixel_value, 1), "(pixel value)")

    return N_all, N_all_non_bgcolor, mean_pixel_value, std_pixel_value



if __name__ == "__main__":
    # Read two input images
    img_in_RGB     = readImage(args[1])

    # Convert RGB image to Grayscale image
    img_in_Gray    = cv2.cvtColor(img_in_RGB,     cv2.COLOR_RGB2GRAY)

    # Extract background color index
    b_index_bgcolor     = img_in_Gray == BGColor_Gray # ndarray(dtype: bool)
    b_index_non_bgcolor = ~b_index_bgcolor

    N_all, N_all_non_bgcolor, mean_pixel_value, std_pixel_value = preProcess(img_in_RGB)
    bin_number              = 100
    ideal_std_pixel_value   = std_pixel_value/4
    ideal_mean_pixel_value  = mean_pixel_value
    img_in_RGB_bgcolor, img_in_RGB_non_bgcolor = separateBackgroundColor()
    pre_processed_img_in_RGB = transformPixelValueDistributionStatistically()
    # adjusted_img_out_RGB    = BrightnessAdjustment(mapped_img_in_RGB)
    # adjusted_img_out_Gray   = cv2.cvtColor(adjusted_img_out_RGB, cv2.COLOR_RGB2GRAY)
    # # Save image
    # adjusted_img_out_BGR = cv2.cvtColor(adjusted_img_out_RGB, cv2.COLOR_RGB2BGR)
    # cv2.imwrite("images/adjusted.bmp", adjusted_img_out_BGR)

    # # Create figure
    # fig = plt.figure(figsize=(6, 8)) # figsize=(width, height)
    # gs  = gridspec.GridSpec(2,1)

    # ax1 = fig.add_subplot(gs[0,0])
    # ax1.set_title('After')
    # ax1.imshow(mapped_img_in_RGB)
    # ax1.set_xticks([]), ax1.set_yticks([])

    # ax2 = fig.add_subplot(gs[1,0])
    # ax2 = grayscaleHist(mapped_img_in_Gray, ax2, "After")
    # ax2.axvline(threshold_pixel_value, color='red')

    # plt.show()


    # # Create figure
    # fig = plt.figure(figsize=(8, 6)) # figsize=(width, height)
    # gs  = gridspec.GridSpec(2,2)

    # ax1 = fig.add_subplot(gs[0,0])
    # ax1.set_title('Before')
    # ax1.imshow(img_in_RGB)
    # ax1.set_xticks([]), ax1.set_yticks([])

    # ax2 = fig.add_subplot(gs[0,1])
    # ax2.set_title('After')
    # ax2.imshow(adjusted_img_RGB)
    # ax2.set_xticks([]), ax2.set_yticks([])

    # ax3 = fig.add_subplot(gs[1,0])
    # ax3 = grayscaleHist(img_in_Gray, ax3, "Before")
    # ax3.axvline(threshold_pixel_value, color='red')

    # ax4 = fig.add_subplot(gs[1,1])
    # ax4 = grayscaleHist(adjusted_img_Gray, ax4, "After")
    # ax4.axvline(threshold_pixel_value, color='red')

    # plt.show()

    # Save image
    # adjusted_img_BGR = cv2.cvtColor(adjusted_img_RGB, cv2.COLOR_RGB2BGR)
    # cv2.imwrite("images/adjusted.bmp", adjusted_img_BGR)

    # # Create figure
    # createFigure(img_in_RGB_L1, _img_RGB, img_adjusted_RGB, ref_pixel_value_L1, ratio_final, max_pixel_value_L1, ratio_of_ref_section_L1)

    # # Save figure and images
    # saveFigureAndImages(p_final, _img_RGB, img_adjusted_RGB)