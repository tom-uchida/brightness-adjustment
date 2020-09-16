######################################
#   @file   adjust_brightness.py
#   @author Tomomasa Uchida
#   @date   2019/02/28
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
import time

# Graph settings
# plt.style.use('seaborn-white')
plt.style.use('bmh')
colors = cycler('color', ['#EE6666', '#3388BB', '#9988DD', '#EECC55', '#88BB44', '#FFBBBB'])
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True, grid=False, prop_cycle=colors)
# plt.rc('grid', color='w', linestyle='solid')
# plt.rc('patch', edgecolor='#E6E6E6')
# plt.rc('lines', linewidth=2)
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["mathtext.rm"] = "Times New Roman"

# Message
print("===================================================")
print("              Brightness Adjustment")
print("                 Tomomasa Uchida")
print("                   2019/02/28")
print("===================================================")

# Check arguments
args = sys.argv
if len(args) != 3:
    print("\n")
    print("USAGE   : $ python adjust_brightness.py [input_image_data] [input_image_data(L=1)]")
    print("EXAMPLE : $ python adjust_brightness.py [input_image.bmp] [input_image_L1.bmp]")
    #raise Exception
    sys.exit()

# Set initial parameter
p_init      = 1.0
p_interval  = 0.01
pct_of_reference_section = 0.01 # 1(%)
# pct_of_reference_section = 0.005 # 1(%)
bgcolor     = 0 # Background color : Black(0, 0, 0)
bin_number  = 100
print("\n")
print("Input image data        (args[1]) :", args[1])
print("Input image data (L=1)  (args[2]) :", args[2])
# print("p_init                           :", p_init)
# print("p_interval                       :", p_interval)
# print("The pct. of reference section       :", pct_of_reference_section*100, "(%)")



def read_image(_img_name):
    # read input image
    img_BGR = cv2.imread(_img_name)

    # convert color BGR to RGB
    img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

    return img_RGB



def create_RGB_histogram(_img_rgb, _ax, _title):
    R_nonzero = _img_rgb[:,:,0][_img_rgb[:,:,0] != bgcolor]
    G_nonzero = _img_rgb[:,:,1][_img_rgb[:,:,1] != bgcolor]
    B_nonzero = _img_rgb[:,:,2][_img_rgb[:,:,2] != bgcolor]
    _ax.hist(R_nonzero.ravel(), bins=bin_number, color='r', alpha=0.5, label="R")
    _ax.hist(G_nonzero.ravel(), bins=bin_number, color='g', alpha=0.5, label="G")
    _ax.hist(B_nonzero.ravel(), bins=bin_number, color='b', alpha=0.5, label="B")
    # _ax.legend()

    _ax.set_title(_title, fontsize=18)
    _ax.set_xlim([-5,260])
    
    return _ax



def create_Grayscale_histogram(_img_gray, _ax, _title):
    img_Gray_nonzero = _img_gray[_img_gray != bgcolor]
    _ax.hist(img_Gray_nonzero.ravel(), bins=bin_number, color='black', alpha=1.0)

    _ax.set_title(_title, fontsize=18)
    _ax.set_xlim([-5,260])
    
    return _ax



# Histograms of Input image(L=1), Input image and Adjusted image
def create_comparative_histogram(_img_in_rgb_L1, _img_in_rgb, _img_out_rgb, _ax, _y_max):
    # Convert RGB to Grayscale
    img_in_Gray_L1             = cv2.cvtColor(_img_in_rgb_L1, cv2.COLOR_RGB2GRAY)
    img_in_Gray_L1_non_bgcolor = img_in_Gray_L1[img_in_Gray_L1 != bgcolor]
    img_in_Gray                = cv2.cvtColor(_img_in_rgb, cv2.COLOR_RGB2GRAY)
    img_in_Gray_non_bgcolor    = img_in_Gray[img_in_Gray != bgcolor]
    img_out_Gray               = cv2.cvtColor(_img_out_rgb, cv2.COLOR_RGB2GRAY)
    img_out_Gray_non_bgcolor   = img_out_Gray[img_out_Gray != bgcolor]
    
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
# End of create_comparative_histogram()



def create_figure(_img_in_RGB_L1, _img_in_RGB, _img_adjusted_RGB, _ref_pixel_value_L1, _pct):
    # Convert RGB to Grayscale
    img_in_Gray_L1     = cv2.cvtColor(_img_in_RGB_L1, cv2.COLOR_RGB2GRAY)
    img_in_Gray        = cv2.cvtColor(_img_in_RGB, cv2.COLOR_RGB2GRAY)
    img_adjusted_Gray  = cv2.cvtColor(_img_adjusted_RGB, cv2.COLOR_RGB2GRAY)

    fig = plt.figure(figsize=(12, 8)) # figsize=(width, height)
    gs  = gridspec.GridSpec(2,3)

    # Input image(L=1)
    ax1 = fig.add_subplot(gs[0,0])
    # ax1.set_title('Input image ($L_{\mathrm{R}}=1$)')
    ax1.set_title('Input image with $L=1$', fontsize=18)
    ax1.imshow(_img_in_RGB_L1)
    ax1.set_xticks([]), ax1.set_yticks([])

    # Input image
    ax2 = fig.add_subplot(gs[0,1])
    ax2.set_title('Input image', fontsize=18)
    ax2.imshow(_img_in_RGB)
    ax2.set_xticks([]), ax2.set_yticks([])

    # adjusted image
    ax3 = fig.add_subplot(gs[0,2])
    ax3.set_title('Adjusted image ($p=$'+str(p_final)+')', fontsize=18)
    ax3.imshow(_img_adjusted_RGB)
    ax3.set_xticks([]), ax3.set_yticks([])

    # Histogram(input image(L=1))
    ax4 = fig.add_subplot(gs[1,0])
    # ax4 = create_Grayscale_histogram(img_in_Gray_L1, ax4, "Input image ($L_{\mathrm{R}}=1$)")
    # ax4 = create_RGB_histogram(_img_in_RGB_L1, ax4, "Input image ($L_{\mathrm{R}}=1$)")
    ax4 = create_RGB_histogram(_img_in_RGB_L1, ax4, "Input image with $L=1$")
    
    # Histogram(input image)
    ax5 = fig.add_subplot(gs[1,1])
    # ax5 = create_Grayscale_histogram(img_in_Gray, ax5, "Input image")
    ax5 = create_RGB_histogram(_img_in_RGB, ax5, "Input image")
    ax5.set_yticks([])

    # Histogram(output image)
    ax6 = fig.add_subplot(gs[1,2])
    # ax6 = create_Grayscale_histogram(img_adjusted_Gray, ax6, "adjusted image")
    ax6 = create_RGB_histogram(_img_adjusted_RGB, ax6, "Adjusted image ($p=$"+str(p_final)+")")
    ax6.set_yticks([])

    # Unify ylim b/w input image and adjusted image
    hist_in_L1,    bins_in_L1    = np.histogram(img_in_Gray_L1[img_in_Gray_L1 != bgcolor],       bin_number)
    hist_in,       bins_in       = np.histogram(img_in_Gray[img_in_Gray != bgcolor],             bin_number)
    hist_adjusted, bins_adjusted = np.histogram(img_adjusted_Gray[img_adjusted_Gray != bgcolor], bin_number)
    list_max = [max(hist_in_L1), max(hist_in), max(hist_adjusted)]
    ax4.set_ylim([0, max(list_max)*1.1])
    ax5.set_ylim([0, max(list_max)*1.1])
    ax6.set_ylim([0, max(list_max)*1.1])

    # # Histograms(Input(L1), Input, adjusted)
    # ax7 = fig.add_subplot(gs[2,:])
    # ax7 = create_comparative_histogram(_img_in_RGB_L1, _img_in_RGB, _img_adjusted_RGB, ax7, max(list_max)*1.1)
    # ax7.set_ylim([0, max(list_max)*1.1])

    # Draw text
    x       = (_ref_pixel_value_L1+max_pixel_value_L1)*0.5 - (255*0.45)
    text    = "["+str(_ref_pixel_value_L1)+", "+str(max_pixel_value_L1)+"]\n→ "+str(round(pct_of_reference_section_L1*100, 2))+"(%)"
    ax4.text(x, max(list_max)*0.5, text, color='black', fontsize=14)
    text    = "["+str(_ref_pixel_value_L1)+", "+str(max_pixel_value_L1)+"]\n→ "+str(round(_pct*100, 2))+"(%)"
    ax6.text(x, max(list_max)*0.5, text, color='black', fontsize=14)

    # Draw reference section
    rect = plt.Rectangle((_ref_pixel_value_L1, 0), max_pixel_value_L1-_ref_pixel_value_L1, max(list_max)*1.1, fc='black', alpha=0.3)
    ax4.add_patch(rect)
    rect = plt.Rectangle((_ref_pixel_value_L1, 0), max_pixel_value_L1-_ref_pixel_value_L1, max(list_max)*1.1, fc='black', alpha=0.3)
    ax6.add_patch(rect)
# End of create_figure()



def calculate_statistics_for_input_image():
    print("Input image (RGB)                :", img_in_RGB.shape) # （height, width, channel）

    # Calc all number of pixels of the input image
    N_all = img_in_RGB.shape[0] * img_in_RGB.shape[1]
    print("N_all                            :", N_all, "(pixels)")

    # Convert RGB to Grayscale
    img_in_Gray             = cv2.cvtColor(img_in_RGB, cv2.COLOR_RGB2GRAY)

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
# End of calculate_statistics_for_input_image()



def calculate_statistics_for_input_image_L1():
    # Convert RGB to Grayscale
    img_in_Gray_L1                 = cv2.cvtColor(img_in_RGB_L1, cv2.COLOR_RGB2GRAY)

    # Exclude background color
    img_in_Gray_non_bgcolor_L1     = img_in_Gray_L1[img_in_Gray_L1 != bgcolor]

    # Calc the number of pixels excluding background color
    N_all_non_bgcolor_L1           = np.sum(img_in_Gray_L1 != bgcolor)

    # Calc max pixel value of the input image (L=1)
    max_pixel_value_L1             = np.max(img_in_Gray_non_bgcolor_L1)
    print("\nMax pixel value (L=1)            :", max_pixel_value_L1, "(pixel value)")

    # Calc mean pixel value (L=1)
    mean_pixel_value_L1            = np.mean(img_in_Gray_non_bgcolor_L1)
    print("Mean pixel value (L=1)           :", round(mean_pixel_value_L1, 1), "(pixel value)")

    # Calc pct. of the max pixel value (L=1)
    num_max_pixel_value_L1         = np.sum(img_in_Gray_non_bgcolor_L1 == max_pixel_value_L1)
    print("Num. of max pixel value (L=1)    :", num_max_pixel_value_L1, "(pixels)")
    pct_max_pixel_value_L1       = num_max_pixel_value_L1 / N_all_non_bgcolor_L1
    # pct_max_pixel_value_L1       = round(pct_max_pixel_value_L1, 8)
    print("The pct. of max pixel value (L=1):", round(pct_max_pixel_value_L1*100, 2), "(%)")

    # Calc most frequent pixel value (L=1)
    bincount = np.bincount(img_in_Gray_non_bgcolor_L1)
    most_frequent_pixel_value_L1   = np.argmax( bincount )
    print("Most frequent pixel value (L=1)  :", most_frequent_pixel_value_L1, "(pixel value)")

    return img_in_Gray_L1, img_in_Gray_non_bgcolor_L1, N_all_non_bgcolor_L1, max_pixel_value_L1, pct_max_pixel_value_L1, 
# End of calculate_statistics_for_input_image_L1()



# Adjust Pixel Value for each RGB
def adjust_pixel_value(_img_RGB, _amp_param):
    adjusted_img_RGB = np.empty((_img_RGB.shape[0], _img_RGB.shape[1], 3), dtype=np.uint8)

    # Apply adjustment
    adjusted_img_RGB[:, :, 0] = cv2.multiply(_img_RGB[:, :, 0], _amp_param) # R
    adjusted_img_RGB[:, :, 1] = cv2.multiply(_img_RGB[:, :, 1], _amp_param) # G
    adjusted_img_RGB[:, :, 2] = cv2.multiply(_img_RGB[:, :, 2], _amp_param) # B

    return adjusted_img_RGB
# End of adjust_pixel_value()



def determine_amplification_factor(_pct_of_reference_section):
    # Initialize
    tmp_pct_of_reference_section = 0.0
    reference_pixel_value_L1       = max_pixel_value_L1

    # Determine reference pixel value in the input image(L=1)
    while tmp_pct_of_reference_section < _pct_of_reference_section:
        # Temporarily calc    
        sum_of_pixels_in_section        = np.sum( (reference_pixel_value_L1 <= img_in_Gray_non_bgcolor_L1) )
        tmp_pct_of_reference_section  = sum_of_pixels_in_section / N_all_non_bgcolor_L1

        # Next pixel value
        reference_pixel_value_L1 -= 1

    reference_pixel_value_L1 += 1
    print("Reference pixel value (L=1)      :", reference_pixel_value_L1, "(pixel value)")
    print("Reference section (L=1)          :", reference_pixel_value_L1, "~", max_pixel_value_L1, "(pixel value)")
    print("The pct. of ref. section (L=1)   :", round(tmp_pct_of_reference_section*100, 2), "(%)")

    # Determine tuning parameter
    p = p_init
    tmp_pct = 0.0
    while tmp_pct < _pct_of_reference_section:
        # Temporarily, adjust pixel value of the input image with p
        tmp_img_adjusted_RGB   = adjust_pixel_value(img_in_RGB, p)
        tmp_img_adjusted_Gray  = cv2.cvtColor(tmp_img_adjusted_RGB, cv2.COLOR_RGB2GRAY)

        # Exclude background color
        tmp_adjusted_img_non_bgcolor_Gray = tmp_img_adjusted_Gray[tmp_img_adjusted_Gray != bgcolor]

        # Then, calc pct of max pixel value(L=1)
        sum_of_pixels_in_reference_section = np.sum(reference_pixel_value_L1 <= tmp_adjusted_img_non_bgcolor_Gray)
        tmp_pct = sum_of_pixels_in_reference_section / N_all_non_bgcolor

        # Update parameter
        p += p_interval

    p_final = round(p, 2)

    return p_final, reference_pixel_value_L1, tmp_pct_of_reference_section
# End of determine_amplification_factor()



def create_adjusted_image(_p_final, _reference_pixel_value_L1):
    print("Amplification factor \"p\"         :", _p_final)

    # Create adjusted image
    img_adjusted_RGB  = adjust_pixel_value(img_in_RGB, _p_final)
    img_adjusted_Gray = cv2.cvtColor(img_adjusted_RGB, cv2.COLOR_RGB2GRAY)

    # Exclude 
    img_adjusted_non_bgcolor_Gray = img_adjusted_Gray[img_adjusted_Gray != bgcolor]

    # For the adjusted image, calc the pct. of num. of pixels in the reference section
    sum_of_pixels_in_reference_section = np.sum( (_reference_pixel_value_L1 <= img_adjusted_Gray) & (img_adjusted_Gray <= max_pixel_value_L1) )
    pct = sum_of_pixels_in_reference_section / N_all_non_bgcolor
    print("The pct. of reference section    :", round(pct*100, 2), "(%)")

    #print("The pct. of num. of pixels to 255   :", round(np.sum(img_adjusted_Gray==255) / N_all_non_bgcolor * 100, 2), "(%)")

    # Create figure
    create_figure(img_in_RGB_L1, img_in_RGB, img_adjusted_RGB, _reference_pixel_value_L1, pct)

    return img_adjusted_RGB
# End of create_adjusted_image()



# Save figure and images
def save_figure_and_images(_p_final, _img_in_RGB, _img_adjusted_RGB):
    fig_name = "IMAGE_DATA/figure_"+str(_p_final)+".png"
    plt.savefig(fig_name)
    # plt.show()

    # convert color RGB to BGR
    img_in_BGR          = cv2.cvtColor(_img_in_RGB,         cv2.COLOR_RGB2BGR)
    img_out_BGR         = cv2.cvtColor(_img_adjusted_RGB,  cv2.COLOR_RGB2BGR)
    input_img_name      = "IMAGE_DATA/input.bmp"
    adjusted_img_name   = "IMAGE_DATA/adjusted_"+str(_p_final)+".bmp"
    cv2.imwrite(input_img_name, img_in_BGR)
    cv2.imwrite(adjusted_img_name, img_out_BGR)

    #exec_command(fig_name, input_img_name, adjusted_img_name)



# Exec. command
def exec_command(_fig_name, _input_img_name, _adjusted_img_name):
    preview_command = ['open', _fig_name, _input_img_name, _adjusted_img_name]
    try:
        res = subprocess.check_call(preview_command)

    except:
        print("ERROR")



if __name__ == "__main__":
    # Read two input images
    img_in_RGB     = read_image(args[1])
    img_in_RGB_L1  = read_image(args[2])

    # Start time count
    start = time.time()

    print("\n")
    print("===================================================")
    print("   Step1. Get max pixel value (L=1)")  
    print("===================================================")
    N_all_non_bgcolor = calculate_statistics_for_input_image()
    img_in_Gray_L1, img_in_Gray_non_bgcolor_L1, N_all_non_bgcolor_L1, max_pixel_value_L1, pct_max_pixel_value_L1 = calculate_statistics_for_input_image_L1()

    print("\n")
    print("===================================================")
    print("   Step2. Search for reference pixel value (L=1)")
    print("===================================================")
    p_final, reference_pixel_value_L1, pct_of_reference_section_L1 = determine_amplification_factor(pct_of_reference_section)

    print("\n")
    print("===================================================")
    print("   Step3. Adjust pixel value")
    print("===================================================")
    adjusted_img_RGB = create_adjusted_image(p_final, reference_pixel_value_L1)

    # End time count
    print ("\nProcessing time                  :", round(time.time() - start, 2),"[sec]")

    # Save figure and images
    save_figure_and_images(p_final, img_in_RGB, adjusted_img_RGB)
