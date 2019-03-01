######################################
#   @file   correct_pixel_value.py
#   @author Tomomasa Uchida
#   @date   2019/02/07
######################################

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cycler
import matplotlib.gridspec as gridspec
import matplotlib.patches as pat
import cv2
import subprocess
import sys
import statistics

# Graph settings
plt.style.use('seaborn-white')
colors = cycler('color', ['#EE6666', '#3388BB', '#9988DD', '#EECC55', '#88BB44', '#FFBBBB'])
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True, grid=False, prop_cycle=colors)
plt.rc('grid', color='w', linestyle='solid')
plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=2)
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["mathtext.rm"] = "Times New Roman"

# Message
print("===================================")
print("     Correct Pixel Value")
print("      author : Tomomasa Uchida")
print("      date   : 2019/02/07")
print("===================================")

# Check arguments
args = sys.argv
if len(args) != 3:
    print("\nUSAGE        : $ python correct_pixel_value.py [input_image_data] [input_image_data(LR=1)]")
    print("Example      : $ python correct_pixel_value.py [input_image.bmp] [input_image_LR1.bmp]]")
    #raise Exception
    sys.exit()

# Set initial parameter
p_init      = 1.0
p_interval  = 0.01
print("\nInput image data (args[1])       :", args[1])
print("Input image data(LR=1) (args[2]) :", args[2])
print("p_init                           :", p_init)
print("p_interval                       :", p_interval)



# Read Input Image
def readImage(_img_name):
    # read input image
    img_BGR = cv2.imread(_img_name)

    # convert color BGR to RGB
    img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

    return img_RGB



# RGB Histogram
def rgbHist(_img_rgb, _ax, _title):
    R_nonzero = _img_rgb[:,:,0][_img_rgb[:,:,0] > 0]
    G_nonzero = _img_rgb[:,:,1][_img_rgb[:,:,1] > 0]
    B_nonzero = _img_rgb[:,:,2][_img_rgb[:,:,2] > 0]
    _ax.hist(R_nonzero.ravel(), bins=50, color='r', alpha=0.5, label="R")
    _ax.hist(G_nonzero.ravel(), bins=50, color='g', alpha=0.5, label="G")
    _ax.hist(B_nonzero.ravel(), bins=50, color='b', alpha=0.5, label="B")
    _ax.legend()

    _ax.set_title('Histogram ('+_title+')')
    _ax.set_xlim([-5,260])
    
    return _ax



# Grayscale Histogram
def grayscaleHist(_img_gray, _ax, _title):
    img_Gray_nonzero = _img_gray[_img_gray > 0]
    _ax.hist(img_Gray_nonzero.ravel(), bins=50, color='black', alpha=1.0)

    _ax.set_title('Histogram ('+_title+')')
    _ax.set_xlim([-5,260])
    
    return _ax



# Histograms of Input image(LR=1), Input image and Corrected image
def comparativeHist(_img_in_rgb_LR1, _img_in_rgb, _img_out_rgb, _ax, _y_max):
    # Convert RGB to Grayscale
    img_in_Gray_LR1         = cv2.cvtColor(_img_in_rgb_LR1, cv2.COLOR_RGB2GRAY)
    img_in_Gray_LR1_nonzero = img_in_Gray_LR1[img_in_Gray_LR1 > 0]
    img_in_Gray             = cv2.cvtColor(_img_in_rgb, cv2.COLOR_RGB2GRAY)
    img_in_Gray_nonzero     = img_in_Gray[img_in_Gray > 0]
    img_out_Gray            = cv2.cvtColor(_img_out_rgb, cv2.COLOR_RGB2GRAY)
    img_out_Gray_nonzero    = img_out_Gray[img_out_Gray > 0]
    
    # input image(LR=1)
    mean_in_LR1 = int(np.mean(img_in_Gray_LR1_nonzero))
    _ax.hist(img_in_Gray_LR1_nonzero.ravel(), bins=50, alpha=0.5, label="Input image ($L_{\mathrm{R}}=1$)", color='#1F77B4')
    _ax.axvline(mean_in_LR1, color='#1F77B4')
    _ax.text(mean_in_LR1+5, _y_max*0.8, "mean:"+str(mean_in_LR1), color='#1F77B4', fontsize='12')

    # input image
    mean_in = int(np.mean(img_in_Gray_nonzero))
    _ax.hist(img_in_Gray_nonzero.ravel(), bins=50, alpha=0.5, label="Input image", color='#FF7E0F')
    _ax.axvline(mean_in, color='#FF7E0F')
    _ax.text(mean_in+5, _y_max*0.6, "mean:"+str(mean_in), color='#FF7E0F', fontsize='12')

    # corrected image
    mean_out = int(np.mean(img_out_Gray_nonzero))
    _ax.hist(img_out_Gray_nonzero.ravel(), bins=50, alpha=0.5, label="Corrected image", color='#2C9F2C')
    _ax.axvline(mean_out, color='#2C9F2C')
    _ax.text(mean_out+5, _y_max*0.7, "mean:"+str(mean_out), color='#2C9F2C', fontsize='12')

    _ax.set_title('Comparative histogram')
    _ax.set_xlabel("Pixel value")
    _ax.set_ylabel("Number of pixels")
    _ax.legend(fontsize='12')
    
    return _ax



# Create Histogram
def createHistogram(_img_in_RGB_LR1, _img_in_RGB, _img_corrected_RGB, _standard_pixel_value_LR1):
    # Convert RGB to Grayscale
    img_in_Gray_LR1     = cv2.cvtColor(_img_in_RGB_LR1, cv2.COLOR_RGB2GRAY)
    img_in_Gray         = cv2.cvtColor(_img_in_RGB, cv2.COLOR_RGB2GRAY)
    img_corrected_Gray  = cv2.cvtColor(_img_corrected_RGB, cv2.COLOR_RGB2GRAY)

    fig = plt.figure(figsize=(10, 6)) # figsize=(width, height)
    gs = gridspec.GridSpec(2,3)

    # Input image(LR=1)
    ax1 = fig.add_subplot(gs[0,0])
    ax1.set_title('Input image ($L_{\mathrm{R}}=1$)')
    ax1.imshow(_img_in_RGB_LR1)
    ax1.set_xticks([]), ax1.set_yticks([])

    # Input image
    ax2 = fig.add_subplot(gs[0,1])
    ax2.set_title('Input image')
    ax2.imshow(_img_in_RGB)
    ax2.set_xticks([]), ax2.set_yticks([])

    # Corrected image
    ax3 = fig.add_subplot(gs[0,2])
    ax3.set_title('Corrected image')
    ax3.imshow(_img_corrected_RGB)
    ax3.set_xticks([]), ax3.set_yticks([])

    # Histogram(input image(LR=1))
    ax4 = fig.add_subplot(gs[1,0])
    # ax4 = grayscaleHist(img_in_Gray_LR1, ax4, "Input image ($L_{\mathrm{R}}=1$)")
    ax4 = rgbHist(_img_in_RGB_LR1, ax4, "Input image ($L_{\mathrm{R}}=1$)")
    
    # Histogram(input image)
    ax5 = fig.add_subplot(gs[1,1])
    # ax5 = grayscaleHist(img_in_Gray, ax5, "Input image")
    ax5 = rgbHist(_img_in_RGB, ax5, "Input image")

    # Histogram(output image)
    ax6 = fig.add_subplot(gs[1,2])
    # ax6 = grayscaleHist(img_corrected_Gray, ax6, "Corrected image")
    ax6 = rgbHist(_img_corrected_RGB, ax6, "Corrected image")

    # Unify ylim b/w input image and corrected image
    hist_in_LR1,    bins_in_LR1     = np.histogram(img_in_Gray_LR1[img_in_Gray_LR1>0],      50)
    hist_in,        bins_in         = np.histogram(img_in_Gray[img_in_Gray>0],              50)
    hist_corrected, bins_corrected  = np.histogram(img_corrected_Gray[img_corrected_Gray>0],50)
    list_max = [max(hist_in_LR1), max(hist_in), max(hist_corrected)]
    ax4.set_ylim([0, max(list_max)*1.1])
    ax5.set_ylim([0, max(list_max)*1.1])
    ax6.set_ylim([0, max(list_max)*1.1])

    # # Histograms(Input(LR1), Input, Corrected)
    # ax7 = fig.add_subplot(gs[2,:])
    # ax7 = comparativeHist(_img_in_RGB_LR1, _img_in_RGB, _img_corrected_RGB, ax7, max(list_max)*1.1)
    # ax7.set_ylim([0, max(list_max)*1.1])

    # Draw text
    x = (_standard_pixel_value_LR1+max_pixel_value_LR1)*0.5 - 100
    text = "["+str(_standard_pixel_value_LR1)+", "+str(max_pixel_value_LR1)+"]\n→ "+str(ratio_of_reference_section*100)+"(%)"
    ax4.text(x, max(list_max)*1.1*0.5, text, color='black', fontsize='12')
    ax6.text(x, max(list_max)*1.1*0.5, text, color='black', fontsize='12')

    # Draw reference section
    rect = plt.Rectangle((_standard_pixel_value_LR1, 0), max_pixel_value_LR1-_standard_pixel_value_LR1, max(list_max)*1.1, fc='black', alpha=0.3)
    ax4.add_patch(rect)
    rect = plt.Rectangle((_standard_pixel_value_LR1, 0), max_pixel_value_LR1-_standard_pixel_value_LR1, max(list_max)*1.1, fc='black', alpha=0.3)
    ax6.add_patch(rect)



# Correct Pixel Value for Each RGB
def correct_pixel_value(_rgb_img, _correct_param):
    corrected_img_RGB = np.empty((_rgb_img.shape[0], _rgb_img.shape[1], 3), dtype=np.uint8)

    # Apply correction
    corrected_img_RGB[:, :, 0] = cv2.multiply(_rgb_img[:, :, 0], _correct_param) # R
    corrected_img_RGB[:, :, 1] = cv2.multiply(_rgb_img[:, :, 1], _correct_param) # G
    corrected_img_RGB[:, :, 2] = cv2.multiply(_rgb_img[:, :, 2], _correct_param) # B

    return corrected_img_RGB



def preProcess():
    print("Input image (RGB)                :", img_in_RGB.shape) # （height, width, channel）

    # Calc all number of pixels of the input image
    N_all = img_in_RGB.shape[0] * img_in_RGB.shape[1]
    print("N_all                            :", N_all, "(pixels)")

    # Calc number of pixels that pixel value is not 0
    img_in_Gray     = cv2.cvtColor(img_in_RGB, cv2.COLOR_RGB2GRAY)
    N_all_nonzero   = np.sum(img_in_Gray > 0)
    print("N_all_nonzero                    :", N_all_nonzero, "(pixels)")

    # Calc max pixel value of the input image(LR=1)
    img_in_Gray_LR1     = cv2.cvtColor(img_in_RGB_LR1, cv2.COLOR_RGB2GRAY)
    N_all_nonzero_LR1   = np.sum(img_in_Gray_LR1 > 0)
    max_pixel_value_LR1 = np.max(img_in_Gray_LR1)
    print("Max pixel value (LR=1)           :", max_pixel_value_LR1, "(pixel value)")

    # Calc ratio of the max pixel value (LR=1)
    num_max_pixel_value_LR1   = np.sum(img_in_Gray_LR1 == max_pixel_value_LR1)
    print("Num. of max pixel value (LR=1)   :", num_max_pixel_value_LR1, "(pixels)")
    ratio_max_pixel_value_LR1 = num_max_pixel_value_LR1 / N_all_nonzero_LR1
    # ratio_max_pixel_value_LR1 = round(ratio_max_pixel_value, 4)
    ratio_max_pixel_value_LR1 = round(ratio_max_pixel_value_LR1, 8)
    print("Ratio of max pixel value (LR=1)  :", round(ratio_max_pixel_value_LR1*100, 2), "(%)")

    # Calc most frequent pixel value (LR=1)
    img_in_Gray_nonzero_LR1         = img_in_Gray_LR1[img_in_Gray_LR1 > 0]
    bincount = np.bincount(img_in_Gray_nonzero_LR1)
    most_frequent_pixel_value_LR1   = np.argmax( bincount )
    print("Most frequent pixel value (LR=1) :", most_frequent_pixel_value_LR1, "(pixel value)")

    return N_all_nonzero, N_all_nonzero_LR1, img_in_Gray_LR1, max_pixel_value_LR1, ratio_max_pixel_value_LR1



def determineCorrectParam(_ratio_of_reference_section):
    print("Ratio of reference section       :", _ratio_of_reference_section*100, "(%)")

    # Determine standard pixel value in the input image(LR=1)
    tmp_reference_section         = 0.0
    standard_pixel_value_LR1      = max_pixel_value_LR1
    while tmp_reference_section < _ratio_of_reference_section:
        # Temporarily calc    
        sum_pixels_in_section     = np.sum( (standard_pixel_value_LR1 <= img_in_Gray_LR1) )
        tmp_reference_section     = sum_pixels_in_section / N_all_nonzero_LR1

        # Next pixel value
        standard_pixel_value_LR1 -= 1

    print("Standard pixel value (LR=1)      :", standard_pixel_value_LR1, "(pixel value)")
    print("Reference section                :", standard_pixel_value_LR1, "~", max_pixel_value_LR1, "(pixel value)")
    print("Ratio of reference section       :", round(tmp_reference_section*100, 2), "(%)")

    # Determine parameter
    p = p_init
    tmp_ratio = 0.0
    while tmp_ratio < _ratio_of_reference_section:
        # Temporarily, correct pixel value of the input image with p
        tmp_corrected_img_RGB   = correct_pixel_value(img_in_RGB, p)
        tmp_corrected_img_Gray  = cv2.cvtColor(tmp_corrected_img_RGB, cv2.COLOR_RGB2GRAY)

        # Then, calc ratio of max pixel value(LR=1)
        tmp_sum_pixels_in_section = np.sum(standard_pixel_value_LR1 <= tmp_corrected_img_Gray)
        tmp_ratio = tmp_sum_pixels_in_section / N_all_nonzero

        # Update parameter
        p += p_interval

    p_final = round(p, 2)

    return p_final, standard_pixel_value_LR1



def correctPixelValue(_p_final, _standard_pixel_value_LR1):
    # Create corrected image
    img_corrected_RGB  = correct_pixel_value(img_in_RGB, _p_final)
    img_corrected_Gray = cv2.cvtColor(img_corrected_RGB, cv2.COLOR_RGB2GRAY)

    print("p_final                          :", _p_final)
    print("Ratio of num. of pixels to 255   :", round(np.sum(img_corrected_Gray==255) / N_all_nonzero * 100, 2), "(%)")

    # Create figure
    createHistogram(img_in_RGB_LR1, img_in_RGB, img_corrected_RGB, _standard_pixel_value_LR1)

    return img_corrected_RGB



# Save figure and images
def saveFigureAndImages(_p_final, _img_in_RGB, _img_corrected_RGB):
    img_in_name = args[1]
    save_path = img_in_name[:-4]

    fig_name = save_path+"_figure_"+str(_p_final)+".png"
    plt.savefig(fig_name)
    # plt.show()

    # convert color RGB to BGR
    # img_in_BGR          = cv2.cvtColor(_img_in_RGB,         cv2.COLOR_RGB2BGR)
    img_out_BGR         = cv2.cvtColor(_img_corrected_RGB,  cv2.COLOR_RGB2BGR)
    # input_img_name      = "images/input.bmp"
    corrected_img_name  = save_path+"_corrected_"+str(_p_final)+".bmp"
    # cv2.imwrite(input_img_name, img_in_BGR)
    cv2.imwrite(corrected_img_name, img_out_BGR)

    execCommand( corrected_img_name )



# Exec. command
def execCommand(_corrected_img_name):
    preview_command = ['open', _corrected_img_name]
    try:
	    res = subprocess.check_call(preview_command)

    except:
	    print("ERROR")



if __name__ == "__main__":
    # Read two input images
    img_in_RGB      = readImage(args[1])
    img_in_RGB_LR1  = readImage(args[2])

    print("\n\n====================================")
    print(" STEP1 : Get max pixel value (LR=1)")  
    print("====================================")
    N_all_nonzero, N_all_nonzero_LR1, img_in_Gray_LR1, max_pixel_value_LR1, ratio_max_pixel_value_LR1 = preProcess()

    print("\n\n================================================")
    print(" STEP2 : Search for standard pixel value (LR=1)")
    print("================================================")
    ratio_of_reference_section = 0.01 # 1(%)
    p_final, standard_pixel_value_LR1 = determineCorrectParam(ratio_of_reference_section)

    print("\n\n=============================")
    print(" STEP3 : Correct pixel value")
    print("=============================")
    img_corrected_RGB = correctPixelValue(p_final, standard_pixel_value_LR1)

    # Save figure and images
    saveFigureAndImages(p_final, img_in_RGB, img_corrected_RGB)