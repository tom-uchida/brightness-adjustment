##########################################################
#   @file   adjust_brightness_decompose_pre-process.py
#   @author Tomomasa Uchida
#   @date   2019/11/14
##########################################################


# ①しきい値輝度値で，低輝度画像と高輝度画像の２つに分解
# ②低輝度画像のみに対して，従来手法を適用し，増幅率pを一時的に決定
# ③高輝度画像の各輝度値に対して，②で求まったpの値で除算する．
# ④低輝度が画像と③で求まった画像を統合
# ⑤従来手法を④で統合した画像に適用する

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
# plt.style.use('seaborn-white')
plt.style.use('bmh')
# colors = cycler('color', ['#EE6666', '#3388BB', '#9988DD', '#EECC55', '#88BB44', '#FFBBBB'])
# plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True, grid=False, prop_cycle=colors)
# plt.rc('grid', color='w', linestyle='solid')
# plt.rc('patch', edgecolor='#E6E6E6')
# plt.rc('lines', linewidth=2)
# plt.rcParams['font.family'] = 'IPAGothic' # font setting
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["mathtext.rm"] = "Times New Roman"

# Message
print("===============================================")
print("     Brightness Adjustment: Decompose ver.")
print("               Tomomasa Uchida")
print("                 2019/11/14")
print("===============================================")

# Check arguments
args = sys.argv
if len(args) != 3:
    print("\n")
    print("USAGE   : $ python adjust_brightness_decompose_pre-process.py [input_image_data] [input_image_data(L=1)]")
    print("EXAMPLE : $ python adjust_brightness_decompose_pre-process.py [input_image.bmp] [input_image_L1.bmp]")
    #raise Exception
    sys.exit()

# Set initial parameter
p_init      = 1.0
p_interval  = 0.01
ratio_of_ref_section = 0.01 # 1(%)
bgcolor     = 0 # Background color : Black(0, 0, 0)
print("\n")
print("Input image data        (args[1]) :", args[1])
print("Input image data (L=1)  (args[2]) :", args[2])
# print("p_init                           :", p_init)
# print("p_interval                       :", p_interval)
# print("Ratio of reference section       :", ratio_of_ref_section*100, "(%)")



# Read Input Image
def readImage(_img_name):
    # read input image
    img_BGR = cv2.imread(_img_name)

    # convert color BGR to RGB
    img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

    return img_RGB



# RGB Histogram
def rgbHist(_img_rgb, _ax, _title):
    R_nonzero = _img_rgb[:,:,0][_img_rgb[:,:,0] != bgcolor]
    G_nonzero = _img_rgb[:,:,1][_img_rgb[:,:,1] != bgcolor]
    B_nonzero = _img_rgb[:,:,2][_img_rgb[:,:,2] != bgcolor]
    _ax.hist(R_nonzero.ravel(), bins=bin_number, color='r', alpha=0.5, label="R")
    _ax.hist(G_nonzero.ravel(), bins=bin_number, color='g', alpha=0.5, label="G")
    _ax.hist(B_nonzero.ravel(), bins=bin_number, color='b', alpha=0.5, label="B")
    _ax.legend()

    _ax.set_title(_title)
    _ax.set_xlim([-5, 260])
    
    return _ax



# Grayscale Histogram
def grayscaleHist(_img_gray, _ax, _title):
    img_Gray_nonzero = _img_gray[_img_gray != bgcolor]
    _ax.hist(img_Gray_nonzero.ravel(), bins=bin_number, color='black', alpha=1.0)

    _ax.set_title(_title)
    _ax.set_xlim([-5, 260])
    
    return _ax



# Histograms of Input image(L=1), Input image and Adjusted image
def comparativeHist(_img_in_rgb_L1, _img_in_rgb, _img_out_rgb, _ax, _y_max):
    # Convert RGB to Grayscale
    img_in_Gray_L1             = cv2.cvtColor(_img_in_rgb_L1, cv2.COLOR_RGB2GRAY)
    img_in_Gray_L1_non_bgcolor = img_in_Gray_L1[img_in_Gray_L1 != bgcolor]
    img_in_Gray                 = cv2.cvtColor(_img_in_rgb, cv2.COLOR_RGB2GRAY)
    img_in_Gray_non_bgcolor     = img_in_Gray[img_in_Gray != bgcolor]
    img_out_Gray                = cv2.cvtColor(_img_out_rgb, cv2.COLOR_RGB2GRAY)
    img_out_Gray_non_bgcolor    = img_out_Gray[img_out_Gray != bgcolor]
    
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



# Create Figure
def createFigure(_img_in_RGB_L1, _img_in_RGB, _img_adjusted_RGB, _ref_pixel_value_L1, _ratio, _max_pixel_value_L1, _ratio_of_ref_section_L1):
    # Convert RGB to Grayscale
    img_in_Gray_L1     = cv2.cvtColor(_img_in_RGB_L1, cv2.COLOR_RGB2GRAY)
    img_in_Gray         = cv2.cvtColor(_img_in_RGB, cv2.COLOR_RGB2GRAY)
    img_adjusted_Gray   = cv2.cvtColor(_img_adjusted_RGB, cv2.COLOR_RGB2GRAY)

    fig = plt.figure(figsize=(10, 6)) # figsize=(width, height)
    gs  = gridspec.GridSpec(2,3)

    # Input image(L=1)
    ax1 = fig.add_subplot(gs[0,0])
    # ax1.set_title('Input image ($L_{\mathrm{R}}=1$)')
    ax1.set_title('Input image ($L=1$)')
    ax1.imshow(_img_in_RGB_L1)
    ax1.set_xticks([]), ax1.set_yticks([])

    # Input image
    ax2 = fig.add_subplot(gs[0,1])
    ax2.set_title('Input image')
    ax2.imshow(_img_in_RGB)
    ax2.set_xticks([]), ax2.set_yticks([])

    # adjusted image
    ax3 = fig.add_subplot(gs[0,2])
    ax3.set_title('Adjusted image')
    ax3.imshow(_img_adjusted_RGB)
    ax3.set_xticks([]), ax3.set_yticks([])

    # Histogram(input image(L=1))
    ax4 = fig.add_subplot(gs[1,0])
    # ax4 = grayscaleHist(img_in_Gray_L1, ax4, "Input image ($L_{\mathrm{R}}=1$)")
    # ax4 = rgbHist(_img_in_RGB_L1, ax4, "Input image ($L_{\mathrm{R}}=1$)")
    ax4 = rgbHist(_img_in_RGB_L1, ax4, "Input image ($L=1$)")
    
    # Histogram(input image)
    ax5 = fig.add_subplot(gs[1,1])
    # ax5 = grayscaleHist(img_in_Gray, ax5, "Input image")
    ax5 = rgbHist(_img_in_RGB, ax5, "Input image")

    # Histogram(output image)
    ax6 = fig.add_subplot(gs[1,2])
    # ax6 = grayscaleHist(img_adjusted_Gray, ax6, "adjusted image")
    ax6 = rgbHist(_img_adjusted_RGB, ax6, "Adjusted image")

    # Unify ylim b/w input image and adjusted image
    hist_in_L1,    bins_in_L1     = np.histogram(img_in_Gray_L1[img_in_Gray_L1 != bgcolor],       bin_number)
    hist_in,       bins_in        = np.histogram(img_in_Gray[img_in_Gray != bgcolor],             bin_number)
    hist_adjusted, bins_adjusted  = np.histogram(img_adjusted_Gray[img_adjusted_Gray != bgcolor], bin_number)
    list_max = [max(hist_in_L1), max(hist_in), max(hist_adjusted)]
    ax4.set_ylim([0, max(list_max)*1.1])
    ax5.set_ylim([0, max(list_max)*1.1])
    ax6.set_ylim([0, max(list_max)*1.1])

    # # Histograms(Input(L1), Input, adjusted)
    # ax7 = fig.add_subplot(gs[2,:])
    # ax7 = comparativeHist(_img_in_RGB_L1, _img_in_RGB, _img_adjusted_RGB, ax7, max(list_max)*1.1)
    # ax7.set_ylim([0, max(list_max)*1.1])

    # Draw text
    x       = (_ref_pixel_value_L1+_max_pixel_value_L1)*0.5 - 100
    text    = "["+str(_ref_pixel_value_L1)+", "+str(_max_pixel_value_L1)+"]\n→ "+str(round(_ratio_of_ref_section_L1*100, 2))+"(%)"
    ax4.text(x, max(list_max)*1.1*0.8, text, color='black', fontsize='12')
    text    = "["+str(_ref_pixel_value_L1)+", "+str(_max_pixel_value_L1)+"]\n→ "+str(round(_ratio*100, 2))+"(%)"
    ax6.text(x, max(list_max)*1.1*0.8, text, color='black', fontsize='12')

    # Draw reference section
    rect = plt.Rectangle((_ref_pixel_value_L1, 0), _max_pixel_value_L1-_ref_pixel_value_L1, max(list_max)*1.1, fc='black', alpha=0.3)
    ax4.add_patch(rect)
    rect = plt.Rectangle((_ref_pixel_value_L1, 0), _max_pixel_value_L1-_ref_pixel_value_L1, max(list_max)*1.1, fc='black', alpha=0.3)
    ax6.add_patch(rect)



# Adjust Pixel Value for each RGB
def adjust_pixel_value(_rgb_img, _adjust_param):
    adjusted_img_RGB = np.empty((_rgb_img.shape[0], _rgb_img.shape[1], 3), dtype=np.uint8)

    # Apply adjustment
    adjusted_img_RGB[:, :, 0] = cv2.multiply(_rgb_img[:, :, 0], _adjust_param) # R
    adjusted_img_RGB[:, :, 1] = cv2.multiply(_rgb_img[:, :, 1], _adjust_param) # G
    adjusted_img_RGB[:, :, 2] = cv2.multiply(_rgb_img[:, :, 2], _adjust_param) # B

    return adjusted_img_RGB



# Search the threshold pixel value
def searchThresholdPixelValue(_img_in_RGB):
    # Convert RGB to Grayscale
    img_in_Gray = cv2.cvtColor(_img_in_RGB, cv2.COLOR_RGB2GRAY)

    # Get histogram of input image
    hist, bins = np.histogram(img_in_Gray[img_in_Gray != bgcolor], bins=bin_number)

    # Convert "tuple" to "numpy array"
    hist = np.array(hist) # print(hist.size)
    bins = np.array(bins) # print(bins)

    # Search a threshold pixel value
    diff_max, index4bins = -1, -1
    for i in range(int(hist.size*0.1), hist.size-1):
        diff = np.abs(hist[i] - hist[i+1])
        # print("diff = ", diff)

        if diff > diff_max:
            diff_max    = diff
            index4bins  = i+1
        # end if
    # end for

    threshold_pixel_value = int(bins[index4bins]) + int(255/bin_number)

    return threshold_pixel_value



# Decompose the input image into two images
def decomposeImage(_img_in_RGB, _threshold_pixel_value):
    print("\nThreshold pixel value            : ", _threshold_pixel_value, "(pixel value)")

    # ndarray(dtype: bool)
    b_index_bgcolor = img_in_Gray == bgcolor
    b_index_low     = (img_in_Gray <= _threshold_pixel_value) & (img_in_Gray != bgcolor)
    b_index_high    = (img_in_Gray  > _threshold_pixel_value) & (img_in_Gray != bgcolor)
    num_bgcolor     = np.count_nonzero(b_index_bgcolor)
    num_low         = np.count_nonzero(b_index_low)
    num_high        = np.count_nonzero(b_index_high)
    print("num_bgcolor  = ", num_bgcolor)
    print("num_low      = ", num_low)
    print("num_high     = ", num_high)
    print("sum          = ", num_bgcolor+num_low+num_high)

    print("\nNumber of \"low\" pixel values\n>", num_low, "/", N_all_non_bgcolor, "(pixels)")
    print(">", round(num_low/N_all_non_bgcolor*100), "(%)")
    print("\nNumber of \"high\" pixel values\n>", num_high, "/", N_all_non_bgcolor, "(pixels)")
    print(">", round(num_high/N_all_non_bgcolor*100), "(%)")

    # Apply decomposition
    low_R  = np.where(b_index_low,  img_in_RGB[:,:,0], bgcolor)
    low_G  = np.where(b_index_low,  img_in_RGB[:,:,1], bgcolor)
    low_B  = np.where(b_index_low,  img_in_RGB[:,:,2], bgcolor)
    high_R = np.where(b_index_high, img_in_RGB[:,:,0], bgcolor)
    high_G = np.where(b_index_high, img_in_RGB[:,:,1], bgcolor)
    high_B = np.where(b_index_high, img_in_RGB[:,:,2], bgcolor)

    # Create low and high pixel value images
    low_img_in_RGB, high_img_in_RGB = img_in_RGB.copy(), img_in_RGB.copy()
    low_img_in_RGB[:,:,0],  low_img_in_RGB[:,:,1],  low_img_in_RGB[:,:,2]  = low_R,  low_G,  low_B
    high_img_in_RGB[:,:,0], high_img_in_RGB[:,:,1], high_img_in_RGB[:,:,2] = high_R, high_G, high_B

    # Convert RGB image to Grayscale image
    low_img_in_Gray  = cv2.cvtColor(low_img_in_RGB,  cv2.COLOR_RGB2GRAY)
    high_img_in_Gray = cv2.cvtColor(high_img_in_RGB, cv2.COLOR_RGB2GRAY)

    # Exclude background color
    low_img_in_Gray_non_bgcolor  = low_img_in_Gray[low_img_in_Gray != bgcolor]
    high_img_in_Gray_non_bgcolor = high_img_in_Gray[high_img_in_Gray != bgcolor]

    # Get min and max pixel value for low and high pixel value images
    low_min   = low_img_in_Gray_non_bgcolor.min()
    low_max   = low_img_in_Gray_non_bgcolor.max()
    print("( low_min, low_max )   = (", low_min, ",", low_max, ")")

    high_min  = high_img_in_Gray_non_bgcolor.min()
    high_max  = high_img_in_Gray_non_bgcolor.max()
    print("( high_min, high_max ) = (", high_min, ",", high_max, ")")


    # # Create figure
    # fig = plt.figure(figsize=(10, 6)) # figsize=(width, height)
    # gs  = gridspec.GridSpec(2,2)

    # # Low image
    # ax1 = fig.add_subplot(gs[0,0])
    # ax1.set_title("Low image")
    # ax1.imshow(low_img_in_RGB)
    # ax1.set_xticks([]), ax1.set_yticks([])

    # # Histogram(Low image)
    # ax2 = fig.add_subplot(gs[1,0])
    # ax2 = grayscaleHist(low_img_in_Gray.ravel(), ax2, "Low image")
    # # ax2.axvline(threshold_pixel_value, color='red')
    # ax2.set_xlim([-5, 260])
    # ax2.set_ylim([0, 20000])

    # # High image
    # ax3 = fig.add_subplot(gs[0,1])
    # ax3.set_title("High image")
    # ax3.imshow(high_img_in_RGB)
    # ax3.set_xticks([]), ax3.set_yticks([])

    # # Histogram(High image)
    # ax4 = fig.add_subplot(gs[1,1])
    # ax4 = grayscaleHist(high_img_in_Gray.ravel(), ax4, "High image")
    # # ax4.axvline(threshold_pixel_value, color='red')
    # ax4.set_xlim([-5, 260])
    # ax4.set_ylim([0, 20000])

    # plt.show()

    return b_index_high, low_img_in_RGB, high_img_in_RGB



def preProcess4HighPixelValueImage(_b_index_high, _low_img_in_RGB, _high_img_in_RGB, _p_tmp):
    tmp_img_uint8 = _high_img_in_RGB.copy()
    tmp_img_float = tmp_img_uint8.astype(float)

    tmp_img_float[:,:,0] = cv2.divide(tmp_img_float[:,:,0], float(_p_tmp))
    tmp_img_float[:,:,1] = cv2.divide(tmp_img_float[:,:,1], float(_p_tmp))
    tmp_img_float[:,:,2] = cv2.divide(tmp_img_float[:,:,2], float(_p_tmp))

    tmp_img_uint8 = tmp_img_float.astype(np.uint8)

    pre_processed_high_img_in_RGB  = _high_img_in_RGB.copy()
    pre_processed_high_img_in_RGB[:,:,0] = np.where(_b_index_high, tmp_img_uint8[:,:,0], bgcolor)
    pre_processed_high_img_in_RGB[:,:,1] = np.where(_b_index_high, tmp_img_uint8[:,:,1], bgcolor)
    pre_processed_high_img_in_RGB[:,:,2] = np.where(_b_index_high, tmp_img_uint8[:,:,2], bgcolor)

    pre_processed_high_img_in_Gray = cv2.cvtColor(pre_processed_high_img_in_RGB, cv2.COLOR_RGB2GRAY)
    pre_processed_high_img_in_Gray_non_bgcolor = pre_processed_high_img_in_Gray[pre_processed_high_img_in_Gray != bgcolor]
    high_min = pre_processed_high_img_in_Gray_non_bgcolor.min()
    high_max = pre_processed_high_img_in_Gray_non_bgcolor.max()
    print("Pre-processed.")
    print("( high_min, high_max ) = (", high_min, ",", high_max, ")")

    # # Create figure
    # fig = plt.figure(figsize=(8, 6)) # figsize=(width, height)
    # gs  = gridspec.GridSpec(2,2)

    # ax1 = fig.add_subplot(gs[0,0])
    # ax1.set_title('Before')
    # ax1.imshow(high_img_in_RGB)
    # ax1.set_xticks([]), ax1.set_yticks([])

    # ax2 = fig.add_subplot(gs[0,1])
    # ax2.set_title('After')
    # ax2.imshow(mapped_high_img_in_RGB)
    # ax2.set_xticks([]), ax2.set_yticks([])

    # ax3 = fig.add_subplot(gs[1,0])
    # ax3 = grayscaleHist(high_img_in_Gray, ax3, "Before")
    # ax3.axvline(threshold_pixel_value, color='red')

    # ax4 = fig.add_subplot(gs[1,1])
    # # ax4 = grayscaleHist(mapped_high_img_in_Gray, ax4, "After")
    # ax4 = rgbHist(mapped_high_img_in_RGB, ax4, "After")
    # ax4.axvline(threshold_pixel_value, color='red')

    # plt.show()

    return pre_processed_high_img_in_RGB



def preProcess(_img_RGB):
    print("Input image (RGB)                :", _img_RGB.shape) # (height, width, channel)

    # Calc all number of pixels of the input image
    N_all = _img_RGB.shape[0] * _img_RGB.shape[1]
    print("N_all                            :", N_all, "(pixels)")

    # Exclude background color
    img_in_Gray_non_bgcolor = img_in_Gray[img_in_Gray != bgcolor]
    
    # Calc the number of pixels excluding background color
    N_all_non_bgcolor       = np.sum(img_in_Gray != bgcolor)
    print("N_all_non_bgcolor                :", N_all_non_bgcolor, "(pixels)")

    # Calc mean pixel value
    max_pixel_value         = np.max(img_in_Gray_non_bgcolor)
    print("Max pixel value                  :", max_pixel_value, "(pixel value)")

    # Calc mean pixel value
    mean_pixel_value        = np.mean(img_in_Gray_non_bgcolor)
    print("Mean pixel value                 :", round(mean_pixel_value, 1), "(pixel value)")

    return N_all_non_bgcolor



def preProcess4L1():
    # Exclude background color
    img_in_Gray_non_bgcolor_L1     = img_in_Gray_L1[img_in_Gray_L1 != bgcolor]

    # Calc the number of pixels excluding background color
    N_all_non_bgcolor_L1           = np.sum(img_in_Gray_L1 != bgcolor)

    # Calc max pixel value of the input image (L=1)
    max_pixel_value_L1             = np.max(img_in_Gray_non_bgcolor_L1)
    print("\nMax pixel value (L=1)           :", max_pixel_value_L1, "(pixel value)")

    # Calc mean pixel value (L=1)
    mean_pixel_value_L1            = np.mean(img_in_Gray_non_bgcolor_L1)
    print("Mean pixel value (L=1)          :", round(mean_pixel_value_L1, 1), "(pixel value)")

    # Calc ratio of the max pixel value (L=1)
    num_max_pixel_value_L1         = np.sum(img_in_Gray_non_bgcolor_L1 == max_pixel_value_L1)
    print("Num. of max pixel value (L=1)   :", num_max_pixel_value_L1, "(pixels)")
    ratio_max_pixel_value_L1       = num_max_pixel_value_L1 / N_all_non_bgcolor_L1
    # ratio_max_pixel_value_L1       = round(ratio_max_pixel_value_L1, 8)
    print("Ratio of max pixel value (L=1)  :", round(ratio_max_pixel_value_L1*100, 2), "(%)")

    # Calc most frequent pixel value (L=1)
    bincount = np.bincount(img_in_Gray_non_bgcolor_L1)
    most_frequent_pixel_value_L1   = np.argmax( bincount )
    print("Most frequent pixel value (L=1) :", most_frequent_pixel_value_L1, "(pixel value)")

    return img_in_Gray_L1, img_in_Gray_non_bgcolor_L1, N_all_non_bgcolor_L1, max_pixel_value_L1, ratio_max_pixel_value_L1, 



def determineAdjustParameter(_img_RGB, _img_in_Gray_non_bgcolor_L1, _N_all_non_bgcolor_L1, _max_pixel_value_L1, _ratio_of_ref_section):
    # Initialize
    tmp_ratio_of_ref_section = 0.0
    ref_pixel_value_L1       = _max_pixel_value_L1

    # Determine reference pixel value in the input image(L=1)
    while tmp_ratio_of_ref_section < _ratio_of_ref_section:
        # Temporarily calc    
        sum_of_pixels_in_section  = np.sum( (ref_pixel_value_L1 <= _img_in_Gray_non_bgcolor_L1) )
        tmp_ratio_of_ref_section  = sum_of_pixels_in_section / _N_all_non_bgcolor_L1

        # Next pixel value
        ref_pixel_value_L1 -= 1

    ref_pixel_value_L1 += 1
    print("Reference pixel value (L=1)     :", ref_pixel_value_L1, "(pixel value)")
    print("Reference section (L=1)         :", ref_pixel_value_L1, "~", _max_pixel_value_L1, "(pixel value)")
    print("Ratio of reference section (L=1):", round(tmp_ratio_of_ref_section*100, 2), "(%)")

    # Determine tuning parameter
    p = p_init
    tmp_ratio = 0.0
    while tmp_ratio < _ratio_of_ref_section:
        # Temporarily, adjust pixel value of the input image with p
        tmp_img_adjusted_RGB   = adjust_pixel_value(_img_RGB, p)
        tmp_img_adjusted_Gray  = cv2.cvtColor(tmp_img_adjusted_RGB, cv2.COLOR_RGB2GRAY)

        # Exclude background color
        tmp_adjusted_img_non_bgcolor_Gray = tmp_img_adjusted_Gray[tmp_img_adjusted_Gray != bgcolor]

        # Then, calc ratio of max pixel value(L=1)
        sum_of_pixels_in_ref_section = np.sum(ref_pixel_value_L1 <= tmp_adjusted_img_non_bgcolor_Gray)
        tmp_ratio = sum_of_pixels_in_ref_section / N_all_non_bgcolor

        # Update parameter
        p += p_interval

    p_final = round(p, 2)

    return p_final, ref_pixel_value_L1, tmp_ratio_of_ref_section



def adjustPixelValue(_img_RGB, _p_final, _ref_pixel_value_L1, _max_pixel_value_L1):
    print("p_final                          :", _p_final)

    # Create adjusted image
    img_adjusted_RGB  = adjust_pixel_value(_img_RGB, _p_final)
    img_adjusted_Gray = cv2.cvtColor(img_adjusted_RGB, cv2.COLOR_RGB2GRAY)

    # Exclude 
    img_adjusted_non_bgcolor_Gray = img_adjusted_Gray[img_adjusted_Gray != bgcolor]

    # For the adjusted image, calc ratio of num. of pixels in the reference section
    sum_of_pixels_in_ref_section = np.sum( (_ref_pixel_value_L1 <= img_adjusted_Gray) & (img_adjusted_Gray <= _max_pixel_value_L1) )
    ratio_final = sum_of_pixels_in_ref_section / N_all_non_bgcolor
    print("Ratio of reference section       :", round(ratio_final*100, 2), "(%)")

    #print("Ratio of num. of pixels to 255   :", round(np.sum(img_adjusted_Gray==255) / N_all_non_bgcolor * 100, 2), "(%)")

    return img_adjusted_RGB, ratio_final



# Save figure and images
def saveFigureAndImages(_p_final, _img_in_RGB, _img_adjusted_RGB):
    fig_name = "images/figure_"+str(_p_final)+".png"
    plt.savefig(fig_name)
    # plt.show()

    # convert color RGB to BGR
    img_in_BGR          = cv2.cvtColor(_img_in_RGB,         cv2.COLOR_RGB2BGR)
    img_out_BGR         = cv2.cvtColor(_img_adjusted_RGB,  cv2.COLOR_RGB2BGR)
    input_img_name      = "images/input.bmp"
    adjusted_img_name   = "images/adjusted_"+str(_p_final)+".bmp"
    cv2.imwrite(input_img_name, img_in_BGR)
    cv2.imwrite(adjusted_img_name, img_out_BGR)

    #execCommand(fig_name, input_img_name, adjusted_img_name)



# Exec. command
def execCommand(_fig_name, _input_img_name, _adjusted_img_name):
    preview_command = ['open', _fig_name, _input_img_name, _adjusted_img_name]
    try:
        res = subprocess.check_call(preview_command)

    except:
        print("ERROR")



def BrightnessAdjustment(_img_RGB):
    print("\n\n====================================")
    print(" STEP1: Get max pixel value (L=1)")  
    print("====================================")
    N_all_non_bgcolor = preProcess(_img_RGB)
    img_in_Gray_L1, img_in_Gray_non_bgcolor_L1, N_all_non_bgcolor_L1, max_pixel_value_L1, ratio_max_pixel_value_L1 = preProcess4L1()

    print("\n\n================================================")
    print(" STEP2: Search for reference pixel value (L=1)")
    print("==================================================")
    p_final, ref_pixel_value_L1, ratio_of_ref_section_L1 = determineAdjustParameter(_img_RGB, img_in_Gray_non_bgcolor_L1, N_all_non_bgcolor_L1, max_pixel_value_L1, ratio_of_ref_section)

    print("\n\n============================")
    print(" STEP3: Adjust pixel value")
    print("============================")
    img_adjusted_RGB, ratio_final = adjustPixelValue(_img_RGB, p_final, ref_pixel_value_L1, max_pixel_value_L1)

    return img_adjusted_RGB, p_final



if __name__ == "__main__":
    # Read two input images
    img_in_RGB     = readImage(args[1])
    img_in_RGB_L1  = readImage(args[2])

    # Convert RGB image to Grayscale image
    img_in_Gray    = cv2.cvtColor(img_in_RGB,     cv2.COLOR_RGB2GRAY)
    img_in_Gray_L1 = cv2.cvtColor(img_in_RGB_L1,  cv2.COLOR_RGB2GRAY)

    # Start time count
    start_time = time.time()

    N_all_non_bgcolor = preProcess(img_in_RGB)
    bin_number = 255
    threshold_pixel_value = searchThresholdPixelValue(img_in_RGB)
    b_index_high, low_img_in_RGB, high_img_in_RGB = decomposeImage(img_in_RGB, threshold_pixel_value)
    adjusted_img_RGB, p_tmp = BrightnessAdjustment(low_img_in_RGB)
    pre_processed_high_img_in_RGB = preProcess4HighPixelValueImage(b_index_high, low_img_in_RGB, high_img_in_RGB, p_tmp)
    pre_processed_img_in_RGB = cv2.scaleAdd(low_img_in_RGB, 1.0, pre_processed_high_img_in_RGB)
    pre_processed_img_in_Gray = cv2.cvtColor(pre_processed_img_in_RGB, cv2.COLOR_RGB2GRAY)
    # adjusted_img_RGB = BrightnessAdjustment(pre_processed_img_in_Gray)
    # adjusted_img_Gray = cv2.cvtColor(adjusted_img_RGB, cv2.COLOR_RGB2GRAY)

    print ("\nElapsed time                     : {0}".format(time.time() - start_time) + "[sec]")

    # Save image
    pre_processed_img_in_BGR = cv2.cvtColor(pre_processed_img_in_RGB, cv2.COLOR_RGB2BGR)
    cv2.imwrite("images/pre-processed_img.bmp", pre_processed_img_in_BGR)

    # # Create figure
    # fig = plt.figure(figsize=(6, 8)) # figsize=(width, height)
    # gs  = gridspec.GridSpec(1,2)

    # ax1 = fig.add_subplot(gs[0,0])
    # ax1.set_title('Before')
    # ax1.imshow(high_img_in_RGB)
    # ax1.set_xticks([]), ax1.set_yticks([])

    # ax2 = fig.add_subplot(gs[0,1])
    # ax2.set_title('After')
    # ax2.imshow(pre_processed_high_img_in_RGB)
    # ax2.set_xticks([]), ax2.set_yticks([])

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
    adjusted_img_BGR = cv2.cvtColor(adjusted_img_RGB, cv2.COLOR_RGB2BGR)
    cv2.imwrite("images/adjusted.bmp", adjusted_img_BGR)

    # # Create figure
    # createFigure(img_in_RGB_L1, _img_RGB, img_adjusted_RGB, ref_pixel_value_L1, ratio_final, max_pixel_value_L1, ratio_of_ref_section_L1)

    # # Save figure and images
    # saveFigureAndImages(p_final, _img_RGB, img_adjusted_RGB)