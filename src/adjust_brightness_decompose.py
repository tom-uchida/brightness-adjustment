###############################################
#   @file   adjust_brightness_decompose.py
#   @author Tomomasa Uchida
#   @date   2019/11/23
###############################################

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
colors = cycler('color', ['#EE6666', '#3388BB', '#9988DD', '#EECC55', '#88BB44', '#FFBBBB'])
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True, grid=False, prop_cycle=colors)
# plt.rc('grid', color='w', linestyle='solid')
# plt.rc('patch', edgecolor='#E6E6E6')
# plt.rc('lines', linewidth=2)
# plt.rcParams['font.family'] = 'IPAGothic' # font setting
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["mathtext.rm"] = "Times New Roman"


# Message
print("=============================================================================")
print("                  Brightness Adjustment: Decompose ver.")
print("                            Tomomasa Uchida")
print("                              2019/11/28")
print("=============================================================================")
print("\n")

# Check arguments
args = sys.argv
if len(args) != 3:
    print("USAGE   : $ python adjust_brightness_decompose.py [input_image_data] [input_image_data(L=1)]")
    print("EXAMPLE : $ python adjust_brightness_decompose.py [input_image.bmp] [input_image_L1.bmp]")
    #raise Exception
    sys.exit()

# Set initial parameter
p_init              = 1.0
p_interval          = 0.01
pct_of_ref_sec4high = 0.01  #  1(%)
pct_of_ref_sec4low  = 0.1   # 10(%)
BGColor             = [0, 0, 0] # Background color
BGColor_Gray        = np.uint8(0.299*BGColor[0]+0.587*BGColor[1]+0.114*BGColor[2])
print("Input image data        (args[1])      :", args[1])
print("Input image data (L=1)  (args[2])      :", args[2])
print("p_init                                 :", p_init)
print("p_interval                             :", p_interval)
print("The pct. of ref. section (high image)  :", pct_of_ref_sec4high*100, "(%)")
print("The pct. of ref. section (low image)   :", pct_of_ref_sec4low*100, "(%)")
print("Background color                       :", BGColor)
print("Background color (Grayscale)           :", BGColor_Gray, "(pixel value)")



def read_image(_img_name):
    img_BGR = cv2.imread(_img_name)
    img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

    return img_RGB
# End of read_image()



def create_RGB_hist(_img_RGB, _ax, _title):
    tmp_b_index_bgcolor = (_img_RGB[:,:,0]==BGColor[0]) & (_img_RGB[:,:,1]==BGColor[1]) & (_img_RGB[:,:,2]==BGColor[2])
    img_R_non_bgcolor = _img_RGB[:,:,0][~tmp_b_index_bgcolor]
    img_G_non_bgcolor = _img_RGB[:,:,1][~tmp_b_index_bgcolor]
    img_B_non_bgcolor = _img_RGB[:,:,2][~tmp_b_index_bgcolor]
    _ax.hist(img_R_non_bgcolor.ravel(), bins=bin_number, color='r', alpha=0.5)
    _ax.hist(img_G_non_bgcolor.ravel(), bins=bin_number, color='g', alpha=0.5)
    _ax.hist(img_B_non_bgcolor.ravel(), bins=bin_number, color='b', alpha=0.5)
    # _ax.legend()

    _ax.set_title(_title, fontsize='14')
    _ax.set_xlim([-5, 260])
    
    return _ax
# End of create_RGB_hist()



def create_Grayscale_hist(_img_Gray, _ax, _title):
    img_Gray_non_bgcolor = _img_Gray[_img_Gray != BGColor_Gray]
    _ax.hist(img_Gray_non_bgcolor.ravel(), bins=bin_number, color='black', alpha=1.0)

    _ax.set_title(_title, fontsize='14')
    _ax.set_xlim([-5, 260])
    
    return _ax
# End of create_Grayscale_hist()



def create_figure_for_high_and_low_pixel_value_images(_fig_name):
    # Create figure
    fig = plt.figure(figsize=(20, 8)) # figsize=(width, height)
    gs  = gridspec.GridSpec(2,5)

    # Input image with L=1
    ax_L1 = fig.add_subplot(gs[0,0])
    ax_L1.set_title("Input image with $L=1$", fontsize='14')
    ax_L1.imshow(img_in_RGB_L1)
    ax_L1.set_xticks([]), ax_L1.set_yticks([])

    # Histogram of the input image with L=1
    ax_L1_hist = fig.add_subplot(gs[1,0])
    ax_L1_hist = create_RGB_hist(img_in_RGB_L1, ax_L1_hist, "Input image with $L=1$")
    ax_L1_hist.set_xlim([-5, 260])
    plt.legend(fontsize='12')

    # Low pixel value image (Original)
    ax_ori_low_img = fig.add_subplot(gs[0,1])
    ax_ori_low_img.set_title("Original low", fontsize='14')
    ax_ori_low_img.imshow(low_img_in_RGB)
    ax_ori_low_img.set_xticks([]), ax_ori_low_img.set_yticks([])

    # Histogram of the low pixel value image (Original)
    ax_ori_low_hist = fig.add_subplot(gs[1,1])
    ax_ori_low_hist = create_RGB_hist(low_img_in_RGB, ax_ori_low_hist, "Original low")
    ax_ori_low_hist.axvline(threshold_pixel_value, color='black', label="threshold")
    ax_ori_low_hist.axvline(mean_pixel_value_low, color='yellow', label="mean")
    ax_ori_low_hist.set_xlim([-5, 260])
    ax_ori_low_hist.set_yticks([])
    plt.legend(fontsize='12')

    # Low pixel value image (Adjusted)
    ax_adj_low_img = fig.add_subplot(gs[0,2])
    ax_adj_low_img.set_title("Adjusted low ($p_{\mathrm{low}}=$"+str(p_low)+")", fontsize='14')
    ax_adj_low_img.imshow(adjusted_low_img_in_RGB)
    ax_adj_low_img.set_xticks([]), ax_adj_low_img.set_yticks([])

    # Histogram of the low pixel value image (Adjusted)
    ax_adj_low_hist = fig.add_subplot(gs[1,2])
    ax_adj_low_hist = create_RGB_hist(adjusted_low_img_in_RGB, ax_adj_low_hist, "Adjusted low ($p_{\mathrm{low}}=$"+str(p_low)+")")
    ax_adj_low_hist.axvline(threshold_pixel_value, color='black', label="threshold")
    ax_adj_low_hist.set_xlim([-5, 260])
    ax_adj_low_hist.set_yticks([])
    plt.legend(fontsize='12')

    # High pixel value image (Original)
    ax_ori_high_img = fig.add_subplot(gs[0,3])
    ax_ori_high_img.set_title("Original high", fontsize='14')
    ax_ori_high_img.imshow(high_img_in_RGB)
    ax_ori_high_img.set_xticks([]), ax_ori_high_img.set_yticks([])

    # Histogram of the high pixel value image (Original)
    ax_ori_high_hist = fig.add_subplot(gs[1,3])
    ax_ori_high_hist = create_RGB_hist(high_img_in_RGB, ax_ori_high_hist, "Original high")
    ax_ori_high_hist.axvline(threshold_pixel_value, color='black', label="threshold")
    ax_ori_high_hist.axvline(mean_pixel_value_high, color='yellow', label="mean")
    ax_ori_high_hist.set_xlim([-5, 260])
    ax_ori_high_hist.set_yticks([])
    plt.legend(fontsize='12')

    # High pixel value image (Adjusted)
    ax_adj_high_img = fig.add_subplot(gs[0,4])
    ax_adj_high_img.set_title("Adjusted high ($p_{\mathrm{high}}=$"+str(p_high)+")", fontsize='14')
    ax_adj_high_img.imshow(adjusted_high_img_in_RGB)
    ax_adj_high_img.set_xticks([]), ax_adj_high_img.set_yticks([])

    # Histogram of the high pixel value image (Adjusted)
    ax_adj_high_hist = fig.add_subplot(gs[1,4])
    ax_adj_high_hist = create_RGB_hist(adjusted_high_img_in_RGB, ax_adj_high_hist, "Adjusted high ($p_{\mathrm{high}}=$"+str(p_high)+")")
    ax_adj_high_hist.axvline(threshold_pixel_value, color='black', label="threshold")
    ax_adj_high_hist.set_xlim([-5, 260])
    ax_adj_high_hist.set_yticks([])
    plt.legend(fontsize='12')

    # Unify value of y-axis
    tmp_b_index_bgcolor = (low_img_in_RGB[:,:,0]==BGColor[0]) & (low_img_in_RGB[:,:,1]==BGColor[1]) & (low_img_in_RGB[:,:,2]==BGColor[2])
    ori_low_R                       = low_img_in_RGB[:,:,0][~tmp_b_index_bgcolor]
    ori_low_G                       = low_img_in_RGB[:,:,1][~tmp_b_index_bgcolor]
    ori_low_B                       = low_img_in_RGB[:,:,2][~tmp_b_index_bgcolor]
    adj_low_R                       = adjusted_low_img_in_RGB[:,:,0][~tmp_b_index_bgcolor]
    adj_low_G                       = adjusted_low_img_in_RGB[:,:,1][~tmp_b_index_bgcolor]
    adj_low_B                       = adjusted_low_img_in_RGB[:,:,2][~tmp_b_index_bgcolor]
    hist_ori_low_R, bins_ori_low_R  = np.histogram(ori_low_R, bin_number)
    hist_ori_low_G, bins_ori_low_G  = np.histogram(ori_low_G, bin_number)
    hist_ori_low_B, bins_ori_low_B  = np.histogram(ori_low_B, bin_number)
    hist_adj_low_R, bins_adj_low_R  = np.histogram(adj_low_R, bin_number)
    hist_adj_low_G, bins_adj_low_G  = np.histogram(adj_low_G, bin_number)
    hist_adj_low_B, bins_adj_low_B  = np.histogram(adj_low_B, bin_number)
    list_hist_ori_low               = [max(hist_ori_low_R), max(hist_ori_low_G), max(hist_ori_low_B)]
    list_hist_adj_low               = [max(hist_adj_low_R), max(hist_adj_low_G), max(hist_adj_low_B)]
    list_hist_max                   = [max(list_hist_ori_low), max(list_hist_adj_low)]
    hist_max                        = max(list_hist_max)
    list_bins_asj_low               = [max(adj_low_R), max(adj_low_G), max(adj_low_B)]
    bins_max                        = max(list_bins_asj_low)
    # print("hist_max: ", hist_max)
    # print("bins_max: ", bins_max)
    ax_ori_low_hist.set_ylim([0, hist_max*1.1])
    ax_adj_low_hist.set_ylim([0, hist_max*1.1])
    ax_ori_high_hist.set_ylim([0, hist_max*1.1])
    ax_adj_high_hist.set_ylim([0, hist_max*1.1])

    # Draw text
    text_L1_low     = str(pct_of_ref_section_L1_low)+"(%)"
    text_L1_high    = str(pct_of_ref_section_L1_high)+"(%)"
    ax_L1_hist.text(left_edge_pixel_value_low-20, hist_max*0.5, text_L1_low, color='black', fontsize='14')
    ax_L1_hist.text(left_edge_pixel_value_high-20, hist_max*0.5, text_L1_high, color='black', fontsize='14')
    text_adj_low    = str(pct_of_ref_section_low)+"(%)"
    text_adj_high   = str(pct_of_ref_section_high)+"(%)"
    ax_adj_low_hist.text(left_edge_pixel_value_low+(bins_max-left_edge_pixel_value_low)*0.4, hist_max*0.5, text_adj_low, color='black', fontsize='14')
    ax_adj_high_hist.text(left_edge_pixel_value_high-50, hist_max*0.5, text_adj_high, color='black', fontsize='14')
    ax_L1_hist.set_ylim([0, hist_max*1.1])

    # Draw reference section
    rect_L1_low     = plt.Rectangle((left_edge_pixel_value_low, 0), 
    right_edge_pixel_value_low-left_edge_pixel_value_low, hist_max*1.1, fc='black', alpha=0.3)
    rect_L1_high    = plt.Rectangle((left_edge_pixel_value_high, 0), right_edge_pixel_value_high-left_edge_pixel_value_high, hist_max*1.1, fc='black', alpha=0.3)
    rect_ori_low    = plt.Rectangle((left_edge_pixel_value_low, 0), 
    right_edge_pixel_value_low-left_edge_pixel_value_low, hist_max*1.1, fc='black', alpha=0.3)
    rect_adj_low    = plt.Rectangle((left_edge_pixel_value_low, 0), 
    bins_max-left_edge_pixel_value_low, hist_max*1.1, fc='black', alpha=0.3)
    rect_ori_high   = plt.Rectangle((left_edge_pixel_value_high, 0), right_edge_pixel_value_high-left_edge_pixel_value_high, hist_max*1.1, fc='black', alpha=0.3)
    rect_adj_high   = plt.Rectangle((left_edge_pixel_value_high, 0), right_edge_pixel_value_high-left_edge_pixel_value_high, hist_max*1.1, fc='black', alpha=0.3)
    ax_L1_hist.add_patch(rect_L1_low), ax_L1_hist.add_patch(rect_L1_high)
    ax_ori_low_hist.add_patch(rect_ori_low)
    ax_adj_low_hist.add_patch(rect_adj_low)
    ax_ori_high_hist.add_patch(rect_ori_high)
    ax_adj_high_hist.add_patch(rect_adj_high)

    plt.savefig(_fig_name)
# End of create_figure_for_high_and_low_pixel_value_images()



def create_figure_for_inputL1_and_input_and_output_images(_fig_name):
    # Create figure
    fig = plt.figure(figsize=(12, 8)) # figsize=(width, height)
    gs  = gridspec.GridSpec(2,3)

    # Input image with L=1
    ax_img_in_L1    = fig.add_subplot(gs[0,0])
    ax_img_in_L1.set_title("Input image with $L=1$", fontsize='14')
    ax_img_in_L1.imshow(img_in_RGB_L1)
    ax_img_in_L1.set_xticks([]), ax_img_in_L1.set_yticks([])

    # Histogram of the input image with L=1
    ax_hist_in_L1   = fig.add_subplot(gs[1,0])
    ax_hist_in_L1   = create_RGB_hist(img_in_RGB_L1, ax_hist_in_L1, "Input image with $L=1$")
    ax_hist_in_L1.set_xlim([-5, 260])
    plt.legend(fontsize='12')

    # Input image
    ax_img_in       = fig.add_subplot(gs[0,1])
    ax_img_in.set_title("Input image", fontsize='14')
    ax_img_in.imshow(img_in_RGB)
    ax_img_in.set_xticks([]), ax_img_in.set_yticks([])

    # Histogram of the input image
    ax_hist_in      = fig.add_subplot(gs[1,1])
    ax_hist_in      = create_RGB_hist(img_in_RGB, ax_hist_in, "Input image")
    ax_hist_in.axvline(threshold_pixel_value, color='black', label="threshold")
    ax_hist_in.axvline(mean_pixel_value, color='yellow', label="mean")
    ax_hist_in.set_xlim([-5, 260])
    ax_hist_in.set_yticks([])
    plt.legend(fontsize='12')

    # Adjust image
    ax_img_out      = fig.add_subplot(gs[0,2])
    ax_img_out.set_title("Adjusted image\n($p_{\mathrm{high}}=$"+str(p_high)+", $p_{\mathrm{low}}=$"+str(p_low)+")", fontsize='14')
    ax_img_out.imshow(adjusted_img_out_RGB)
    ax_img_out.set_xticks([]), ax_img_out.set_yticks([])

    # Histogram of the adjusted image
    ax_hist_out     = fig.add_subplot(gs[1,2])
    ax_hist_out     = create_RGB_hist(adjusted_img_out_RGB, ax_hist_out, "Adjusted image\n($p_{\mathrm{high}}=$"+str(p_high)+", $p_{\mathrm{low}}=$"+str(p_low)+")")
    ax_hist_out.axvline(threshold_pixel_value, color='black', label="threshold")
    ax_hist_out.axvline(mean_pixel_value_adjusted, color='yellow', label="mean")
    ax_hist_out.set_xlim([-5, 260])
    ax_hist_out.set_yticks([])
    plt.legend(fontsize='12')

    # Unify value of y-axis
    tmp_b_index_bgcolor = (img_in_RGB[:,:,0]==BGColor[0])&(img_in_RGB[:,:,1]==BGColor[1])&(img_in_RGB[:,:,2]==BGColor[2])
    in_R                  = img_in_RGB[:,:,0][~tmp_b_index_bgcolor]
    in_G                  = img_in_RGB[:,:,1][~tmp_b_index_bgcolor]
    in_B                  = img_in_RGB[:,:,2][~tmp_b_index_bgcolor]
    hist_in_R, bins_in_R  = np.histogram(in_R, bin_number)
    hist_in_G, bins_in_G  = np.histogram(in_G, bin_number)
    hist_in_B, bins_in_B  = np.histogram(in_B, bin_number)
    list_hist_in          = [max(hist_in_R), max(hist_in_G), max(hist_in_B)]
    hist_max              = max(list_hist_in)
    ax_hist_in_L1.set_ylim([0, hist_max*1.1])
    ax_hist_in.set_ylim([0, hist_max*1.1])
    ax_hist_out.set_ylim([0, hist_max*1.1])

    # Draw text
    text_in_L1_low     = str(pct_of_ref_section_L1_low)+"(%)"
    text_in_L1_high    = str(pct_of_ref_section_L1_high)+"(%)"
    ax_hist_in_L1.text(left_edge_pixel_value_low-20, hist_max*0.5, text_in_L1_low, color='black', fontsize='14')
    ax_hist_in_L1.text(left_edge_pixel_value_high-20, hist_max*0.5, text_in_L1_high, color='black', fontsize='14')

    # Draw reference section
    rect_in_L1_low        = plt.Rectangle((left_edge_pixel_value_low, 0), 
    right_edge_pixel_value_low-left_edge_pixel_value_low, hist_max*1.1, fc='black', alpha=0.3)
    rect_in_L1_high       = plt.Rectangle((left_edge_pixel_value_high, 0), right_edge_pixel_value_high-left_edge_pixel_value_high, hist_max*1.1, fc='black', alpha=0.3)
    ax_hist_in_L1.add_patch(rect_in_L1_low), ax_hist_in_L1.add_patch(rect_in_L1_high)
    
    plt.savefig(_fig_name)
# End of create_figure_for_inputL1_and_input_and_output_images()


def calculate_statistics_for_input_image():
    print("Input image (RGB)                      :", img_in_RGB.shape) # (height, width, channel)

    # Calc all number of pixels of the input image
    N_all = img_in_RGB.shape[0] * img_in_RGB.shape[1]
    print("N_all                                  :", N_all, "(pixels)")

    # Exclude background color
    img_in_Gray_non_bgcolor = img_in_Gray[img_in_Gray != BGColor_Gray]
    
    # Calc the number of pixels excluding background color
    N_all_non_bgcolor       = np.sum(img_in_Gray != BGColor_Gray)
    print("N_all_non_bgcolor                      :", N_all_non_bgcolor, "(pixels)")

    # Calc mean pixel value
    max_pixel_value         = np.max(img_in_Gray_non_bgcolor)
    print("Max pixel value                        :", max_pixel_value, "(pixel value)")

    # Calc mean pixel value
    mean_pixel_value        = np.uint8(np.mean(img_in_Gray_non_bgcolor))
    print("Mean pixel value                       :", mean_pixel_value, "(pixel value)")

    # Calc std pixel value
    std_pixel_value         = np.uint8(np.std(img_in_Gray_non_bgcolor))
    print("Std pixel value                        :", std_pixel_value, "(pixel value)")

    return N_all_non_bgcolor, mean_pixel_value, std_pixel_value
# End of calculate_statistics_for_input_image()



def calculate_statistics_for_input_image_L1():
    # Exclude background color
    img_in_Gray_non_bgcolor_L1     = img_in_Gray_L1[img_in_Gray_L1 != BGColor_Gray]

    # Calc the number of pixels excluding background color
    N_all_non_bgcolor_L1           = np.sum(img_in_Gray_L1 != BGColor_Gray)

    # Calc max pixel value of the input image (L=1)
    max_pixel_value_L1             = np.max(img_in_Gray_non_bgcolor_L1)
    print("\nMax pixel value (L=1)                  :", max_pixel_value_L1, "(pixel value)")

    # Calc mean pixel value (L=1)
    mean_pixel_value_L1            = np.uint8(np.mean(img_in_Gray_non_bgcolor_L1))
    print("Mean pixel value (L=1)                 :", round(mean_pixel_value_L1, 1), "(pixel value)")

    # Calc the pct. of the max pixel value (L=1)
    num_max_pixel_value_L1         = np.sum(img_in_Gray_non_bgcolor_L1 == max_pixel_value_L1)
    print("Num. of max pixel value (L=1)          :", num_max_pixel_value_L1, "(pixels)")
    pct_max_pixel_value_L1       = num_max_pixel_value_L1 / N_all_non_bgcolor_L1
    # pct_max_pixel_value_L1       = round(pct_max_pixel_value_L1, 8)
    print("The pct. of max pixel value (L=1)      :", round(pct_max_pixel_value_L1*100, 2), "(%)")

    # Calc most frequent pixel value (L=1)
    bincount = np.bincount(img_in_Gray_non_bgcolor_L1)
    most_frequent_pixel_value_L1   = np.argmax( bincount )
    print("Most frequent pixel value (L=1)        :", most_frequent_pixel_value_L1, "(pixel value)")

    return N_all_non_bgcolor_L1, max_pixel_value_L1
# End of calculate_statistics_for_input_image_L1()



# Decompose the input image into two images (high pixel value image and low pixel value image)
def decompose_input_image(_threshold_pixel_value):
    print("Threshold pixel value                  :", _threshold_pixel_value, "(pixel value)")

    # ndarray(dtype: bool)
    # b_index_bgcolor = (img_in_RGB[:,:,0]==BGColor[0]) & (img_in_RGB[:,:,1]==BGColor[1]) & (img_in_RGB[:,:,2]==BGColor[2])
    b_index_high    = (img_in_Gray  > _threshold_pixel_value) & (~b_index_bgcolor)
    b_index_low     = (img_in_Gray <= _threshold_pixel_value) & (~b_index_bgcolor)
    N_bgcolor       = np.count_nonzero(b_index_bgcolor)
    N_high, N_low   = np.count_nonzero(b_index_high), np.count_nonzero(b_index_low)
    print("The pct. of high pixel values          :", round(N_high/N_all_non_bgcolor*100),   "(%)")
    print("The pct. of low pixel values           :", round(N_low/N_all_non_bgcolor*100),    "(%)")

    # Apply decomposition and create low and high pixel value images
    high_img_in_RGB, low_img_in_RGB = img_in_RGB.copy(), img_in_RGB.copy()
    high_img_in_RGB[:,:,0] = np.where(b_index_high, img_in_RGB[:,:,0], BGColor[0])
    high_img_in_RGB[:,:,1] = np.where(b_index_high, img_in_RGB[:,:,1], BGColor[1])
    high_img_in_RGB[:,:,2] = np.where(b_index_high, img_in_RGB[:,:,2], BGColor[2])
    low_img_in_RGB[:,:,0]  = np.where(b_index_low,  img_in_RGB[:,:,0], BGColor[0])
    low_img_in_RGB[:,:,1]  = np.where(b_index_low,  img_in_RGB[:,:,1], BGColor[1])
    low_img_in_RGB[:,:,2]  = np.where(b_index_low,  img_in_RGB[:,:,2], BGColor[2])

    # Convert RGB image to Grayscale image
    high_img_in_Gray = cv2.cvtColor(high_img_in_RGB, cv2.COLOR_RGB2GRAY)
    low_img_in_Gray  = cv2.cvtColor(low_img_in_RGB,  cv2.COLOR_RGB2GRAY)

    # Calulate mean pixel value
    mean_pixel_value_high  = np.uint8(np.mean(high_img_in_Gray[high_img_in_Gray != BGColor_Gray]))
    mean_pixel_value_low   = np.uint8(np.mean(low_img_in_Gray[low_img_in_Gray   != BGColor_Gray]))
    print("Mean pixel value (high image)          :", mean_pixel_value_high, "(pixel value)")
    print("Mean pixel value (low image)           :", mean_pixel_value_low,  "(pixel value)")

    return high_img_in_RGB, low_img_in_RGB, N_high, N_low, mean_pixel_value_high, mean_pixel_value_low
# End of decompose_input_image()



# Adjust pixel value for each RGB
def tmp_adjust_pixel_value(_tmp_img_RGB, _img_RGB, _amplification_factor):
    # tmp_adjusted_img_RGB = np.empty((_img_RGB.shape[0], _img_RGB.shape[1], 3), dtype=np.uint8)

    # Apply adjustment
    _tmp_img_RGB[:,:,0] = cv2.multiply(_img_RGB[:,:,0], _amplification_factor) # R
    _tmp_img_RGB[:,:,1] = cv2.multiply(_img_RGB[:,:,1], _amplification_factor) # G
    _tmp_img_RGB[:,:,2] = cv2.multiply(_img_RGB[:,:,2], _amplification_factor) # B

    return _tmp_img_RGB

    # tmp_img_RGB = _img_RGB * _amplification_factor
    # return tmp_img_RGB
# End of tmp_adjust_pixel_value()



def determine_amplification_factor(_img_RGB, _right_edge_pixel_value, _pct_of_ref_section, _N_all_non_bgcolor):
    print("Theoretical pct. of ref. section (L=1) :", _pct_of_ref_section*100, "(%)")

    # Initialize
    tmp_pct_of_ref_section_L1    = 0.0
    tmp_left_edge_pixel_value_L1 = _right_edge_pixel_value

    # NOTE: For the input image with L=1
    # Determine left edge pixel value in the input image with L=1
    img_in_Gray_L1_non_bgcolor   = img_in_Gray_L1[img_in_Gray_L1 != BGColor_Gray]
    while tmp_pct_of_ref_section_L1 <= _pct_of_ref_section:
        # Temporarily, calculate the percentage of pixels in the reference section
        tmp_num_of_pixels         = (tmp_left_edge_pixel_value_L1 <= img_in_Gray_L1_non_bgcolor) & (img_in_Gray_L1_non_bgcolor <= _right_edge_pixel_value)
        tmp_pct_of_ref_section_L1 = np.sum( tmp_num_of_pixels ) / N_all_non_bgcolor_L1

        # Next pixel value
        tmp_left_edge_pixel_value_L1 -= 1
    # end while

    left_edge_pixel_value_L1    = tmp_left_edge_pixel_value_L1 - 1
    pct_of_ref_section_L1       = round(tmp_pct_of_ref_section_L1*100, 1)
    # print("Left edge pixel value (L=1)         :", left_edge_pixel_value_L1, "(pixel value)")
    # print("Right edge pixel value (L=1)        :", _right_edge_pixel_value, "(pixel value)")
    print("Reference section (L=1)                :", "[", left_edge_pixel_value_L1, ",", _right_edge_pixel_value, "]", "(pixel value)")
    print("Actual pct. of ref. section (L=1)      :", pct_of_ref_section_L1, "(%)")


    # NOTE: For the input image
    # Determine amplification factor "p" in the input image
    tmp_p                       = p_init
    tmp_pct_of_ref_section      = 0.0
    tmp_img_RGB                 = np.empty((img_in_RGB.shape[0], img_in_RGB.shape[1], 3), dtype=np.uint8)
    while tmp_pct_of_ref_section <= _pct_of_ref_section:
        # Temporarily, adjust pixel value of the input image with p
        tmp_adjusted_img_RGB    = tmp_adjust_pixel_value(tmp_img_RGB, _img_RGB, tmp_p)
        tmp_adjusted_img_Gray   = cv2.cvtColor(tmp_adjusted_img_RGB, cv2.COLOR_RGB2GRAY)

        # Exclude background color
        tmp_adjusted_img_Gray_non_bgcolor = tmp_adjusted_img_Gray[tmp_adjusted_img_Gray != BGColor_Gray]

        # Calculate the percentage of max pixel value
        # tmp_num_of_pixels = (left_edge_pixel_value_L1 <= tmp_adjusted_img_Gray_non_bgcolor) & (tmp_adjusted_img_Gray_non_bgcolor <= _right_edge_pixel_value)
        tmp_num_of_pixels       = (left_edge_pixel_value_L1 <= tmp_adjusted_img_Gray_non_bgcolor)
        tmp_pct_of_ref_section  = np.sum( tmp_num_of_pixels ) / _N_all_non_bgcolor

        # Update parameter
        tmp_p += p_interval
    # end while

    p_final                     = round((tmp_p - p_interval), 2)
    pct_of_ref_section          = round(tmp_pct_of_ref_section*100, 1)
    print("\nDetermined amplification factor \"p\"    :", p_final)

    return p_final, left_edge_pixel_value_L1, pct_of_ref_section_L1, pct_of_ref_section
# End of determine_amplification_factor()



def adjust_pixel_value(_img_RGB, _p_final, _left_edge_pixel_value_L1, _right_edge_pixel_value, _N_all_non_bgcolor):
    # Create adjusted image
    tmp_img_RGB       = np.empty((img_in_RGB.shape[0], img_in_RGB.shape[1], 3), dtype=np.uint8)
    adjusted_img_RGB  = tmp_adjust_pixel_value(tmp_img_RGB, _img_RGB, _p_final)
    adjusted_img_Gray = cv2.cvtColor(adjusted_img_RGB, cv2.COLOR_RGB2GRAY)

    # Exclude background color
    adjusted_img_Gray_non_bgcolor = adjusted_img_Gray[adjusted_img_Gray != BGColor_Gray]

    # NOTE: For the adjusted image
    # Calculate the percentage of number of pixels in the reference section
    # tmp_num_of_pixels = (_left_edge_pixel_value_L1 <= adjusted_img_Gray_non_bgcolor) & (adjusted_img_Gray_non_bgcolor <= _right_edge_pixel_value)
    tmp_num_of_pixels           = _left_edge_pixel_value_L1 <= adjusted_img_Gray_non_bgcolor
    final_pct_of_ref_section    = np.sum( tmp_num_of_pixels ) / _N_all_non_bgcolor
    print("Final pct. of ref. section             :", round(final_pct_of_ref_section*100, 1), "(%)")

    return adjusted_img_RGB
# End of adjust_pixel_value()



def brightness_adjustment(_img_RGB, _right_edge_pixel_value, _pct_of_ref_section, _N_all):
    # Determine amplification factor "p"
    p_final, left_edge_pixel_value_L1, pct_of_ref_section_L1, pct_of_ref_section = determine_amplification_factor(_img_RGB, _right_edge_pixel_value, _pct_of_ref_section, _N_all)
    
    # Adjust brightness of the image
    adjusted_img_RGB = adjust_pixel_value(_img_RGB, p_final, left_edge_pixel_value_L1, _right_edge_pixel_value, _N_all)

    return adjusted_img_RGB, p_final, left_edge_pixel_value_L1, pct_of_ref_section_L1, pct_of_ref_section
# End of brightness_adjustment()



def save_adjusted_image(_adjusted_img_out_RGB, _p_high, _p_low):
    # Save input image
    input_img_name      = "IMAGE_DATA/input.bmp"
    cv2.imwrite(input_img_name,     cv2.cvtColor(img_in_RGB, cv2.COLOR_RGB2BGR))

    # Save adjusted image
    adjusted_img_name   = "IMAGE_DATA/adjusted_phigh"+str(_p_high)+"_plow"+str(_p_low)+".bmp"
    cv2.imwrite(adjusted_img_name,  cv2.cvtColor(_adjusted_img_out_RGB, cv2.COLOR_RGB2BGR))

    # NOTE: macOS only
    # Exec. "open" command
    exec_open_command(fig_name_1, fig_name_2, input_img_name, adjusted_img_name)
# End of save_adjusted_image()



# Exec. open command
def exec_open_command(_fig_name_1, _fig_name_2, _input_img_name, _adjusted_img_name):
    open_command = ['open', _fig_name_1, _fig_name_2, _input_img_name, _adjusted_img_name]

    try:
        res = subprocess.check_call(open_command)

    except:
        print("\"open\" command error.")
# End of exec_open_command()



if __name__ == "__main__":
    # Read two input images
    img_in_RGB     = read_image(args[1])
    img_in_RGB_L1  = read_image(args[2])

    # Convert RGB image to Grayscale image
    img_in_Gray    = cv2.cvtColor(img_in_RGB,     cv2.COLOR_RGB2GRAY)
    img_in_Gray_L1 = cv2.cvtColor(img_in_RGB_L1,  cv2.COLOR_RGB2GRAY)

    # Get indexes of background color pixel
    b_index_bgcolor = (img_in_RGB[:,:,0]==BGColor[0]) & (img_in_RGB[:,:,1]==BGColor[1]) & (img_in_RGB[:,:,2]==BGColor[2])

    # Start time count
    start_time     = time.time()

    # Calculate statistics for two input images
    N_all_non_bgcolor, mean_pixel_value, std_pixel_value = calculate_statistics_for_input_image()
    N_all_non_bgcolor_L1, max_pixel_value_L1             = calculate_statistics_for_input_image_L1()

    print("\n")
    print("=============================================================================")
    print("   Step1. Decompose the input image to \"high\" and \"low\" pixel value images")
    print("=============================================================================")
    bin_number                      = 50
    # threshold_pixel_value           = np.uint8(mean_pixel_value + 2*std_pixel_value)
    threshold_pixel_value           = np.uint8(mean_pixel_value + 1*std_pixel_value)
    # threshold_pixel_value           = np.uint8(mean_pixel_value)
    high_img_in_RGB, low_img_in_RGB, N_high, N_low, mean_pixel_value_high, mean_pixel_value_low = decompose_input_image(threshold_pixel_value)

    print("\n")
    print("=============================================================================")
    print("   Step2. Adjust brightness of the \"high\" pixel value image")
    print("=============================================================================")
    right_edge_pixel_value_high     = max_pixel_value_L1
    adjusted_high_img_in_RGB, p_high, left_edge_pixel_value_high, pct_of_ref_section_L1_high, pct_of_ref_section_high         = brightness_adjustment(high_img_in_RGB, right_edge_pixel_value_high, pct_of_ref_sec4high, N_high)

    print("\n")
    print("=============================================================================")
    print("   Step3. Adjust brightness of the \"low\" pixel value image")
    print("=============================================================================")
    right_edge_pixel_value_low      = mean_pixel_value_high
    # right_edge_pixel_value_low      = mean_pixel_value + 3*std_pixel_value
    adjusted_low_img_in_RGB, p_low, left_edge_pixel_value_low, pct_of_ref_section_L1_low, pct_of_ref_section_low          = brightness_adjustment(low_img_in_RGB, right_edge_pixel_value_low, pct_of_ref_sec4low, N_low)

    print("\n")
    print("=============================================================================")
    print("   Step4. Resynthesis \"high\" and \"low\" pixel value images")
    print("=============================================================================")
    adjusted_img_out_RGB            = cv2.scaleAdd(adjusted_high_img_in_RGB, 1.0, adjusted_low_img_in_RGB)
    adjusted_img_out_Gray           = cv2.cvtColor(adjusted_img_out_RGB, cv2.COLOR_RGB2GRAY)
    mean_pixel_value_adjusted       = np.mean(adjusted_img_out_Gray[adjusted_img_out_Gray != BGColor_Gray])

    # End time count
    end_time                        = round(time.time()-start_time, 2)

    # Create and save two figures
    fig_name_1 = "IMAGE_DATA/fig_high_low_images.png"
    fig_name_2 = "IMAGE_DATA/fig_inputL1_input_adjusted_images.png"
    create_figure_for_high_and_low_pixel_value_images(fig_name_1)
    create_figure_for_inputL1_and_input_and_output_images(fig_name_2)

    # Save images
    save_adjusted_image(adjusted_img_out_RGB, p_high, p_low)

    print("Amplification factor \"p_high\"          :", p_high)
    print("Amplification factor \"p_low\"           :", p_low)
    print("\nProcessing time                        :", end_time, "[sec]")