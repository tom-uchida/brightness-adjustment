import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cycler
import matplotlib.gridspec as gridspec
import matplotlib.patches as pat
import cv2
import subprocess
import sys
import statistics

plt.style.use('seaborn-white')

args = sys.argv
if len(args) != 3:
    raise Exception('\nUSAGE\n> $ python auto_correct_pixel_value.py [input_image_data] [input_image_data(LR=1)]')
    raise Exception('\n\nFor example\n> $ python auto_correct_pixel_value.py [input_image.bmp] [input_image_LR1.bmp]]\n')
    sys.exit()

colors = cycler('color', ['#EE6666', '#3388BB', '#9988DD', '#EECC55', '#88BB44', '#FFBBBB'])
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True, grid=False, prop_cycle=colors)
plt.rc('grid', color='w', linestyle='solid')
plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=2)

plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["mathtext.rm"] = "Times New Roman"



# Set initial parameter
p_init = 1.0
p_interval = 0.01
print("\n===== Initial parameter =====")
print("input_image_data\n>",            args[1], "(args[1])")
print("\ninput_image_data(LR=1)\n>",    args[2], "(args[2])")
print("\np_init\n>",                    p_init)
print("\np_interval\n>",                p_interval)



# Read Input Image
def read_img(_img_name):
    # read input image
    img_BGR = cv2.imread(_img_name)

    # convert color (BGR → RGB)
    img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

    return img_RGB



# RGB histogram
def rgb_hist(_img_rgb, _ax, _title):
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



# Grayscale histogram
def grayscale_hist(_img_gray, _ax, _title):
    img_Gray_nonzero = _img_gray[_img_gray > 0]
    _ax.hist(img_Gray_nonzero.ravel(), bins=50, color='black', alpha=1.0)

    _ax.set_title('Histogram ('+_title+')')
    _ax.set_xlim([-5,260])
    
    return _ax



# Histograms of Input(LR=1), Input and Corrected
def comparative_hist(_img_in_rgb_LR1, _img_in_rgb, _img_out_rgb, _ax, _y_max):
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



def plot_histogram(_img_in_RGB_LR1, _img_in_RGB, _img_corrected_RGB, _standard_pixel_value_LR1=None, _median_bw_standard_255_LR1=None):
    # Convert RGB to Grayscale
    img_in_Gray_LR1     = cv2.cvtColor(_img_in_RGB_LR1, cv2.COLOR_RGB2GRAY)
    img_in_Gray         = cv2.cvtColor(_img_in_RGB, cv2.COLOR_RGB2GRAY)
    img_corrected_Gray  = cv2.cvtColor(_img_corrected_RGB, cv2.COLOR_RGB2GRAY)

    fig = plt.figure(figsize=(10, 6)) # (width, height)
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
    # ax4 = grayscale_hist(img_in_Gray_LR1, ax4, "Input image ($L_{\mathrm{R}}=1$)")
    ax4 = rgb_hist(_img_in_RGB_LR1, ax4, "Input image ($L_{\mathrm{R}}=1$)")
    
    # Histogram(input image)
    ax5 = fig.add_subplot(gs[1,1])
    # ax5 = grayscale_hist(img_in_Gray, ax5, "Input image")
    ax5 = rgb_hist(_img_in_RGB, ax5, "Input image")

    # Histogram(output image)
    ax6 = fig.add_subplot(gs[1,2])
    # ax6 = grayscale_hist(img_corrected_Gray, ax6, "Corrected image")
    ax6 = rgb_hist(_img_corrected_RGB, ax6, "Corrected image")

    # Unify ylim b/w input image and corrected image
    hist_in_LR1, bins_in_LR1 = np.histogram(img_in_Gray_LR1[img_in_Gray_LR1>0], 50)
    hist_in,     bins_in     = np.histogram(img_in_Gray[img_in_Gray>0],         50)
    hist_corrected,    bins_corrected    = np.histogram(img_corrected_Gray[img_corrected_Gray>0], 50)
    list_max = [max(hist_in_LR1), max(hist_in), max(hist_corrected)]
    ax4.set_ylim([0, max(list_max)*1.1])
    ax5.set_ylim([0, max(list_max)*1.1])
    ax6.set_ylim([0, max(list_max)*1.1])

    # # Histograms(Input(LR1), Input, Corrected)
    # ax7 = fig.add_subplot(gs[2,:])
    # ax7 = comparative_hist(_img_in_RGB_LR1, _img_in_RGB, _img_corrected_RGB, ax7, max(list_max)*1.1)
    # ax7.set_ylim([0, max(list_max)*1.1])

    # If most frequent value is 255 (LR=1)
    if _median_bw_standard_255_LR1 is not None:
        # Draw line
        ax4.axvline(_median_bw_standard_255_LR1, color='black', alpha=0.5)
        ax4.text(_median_bw_standard_255_LR1-110, max(list_max)*1.1*0.7, "median:"+str(_median_bw_standard_255_LR1), color='black', fontsize='12')

        # Draw rectangle
        rect = plt.Rectangle((_standard_pixel_value_LR1, 0), 254-_standard_pixel_value_LR1, max(list_max)*1.1, fc='black', alpha=0.3)
        ax4.add_patch(rect)

    # If max pixel value is less than 255 (LR=1)
    elif _standard_pixel_value_LR1 is not None:
        # Draw text
        x = (_standard_pixel_value_LR1+max_pixel_value_LR1)*0.5 - 100
        text = "["+str(_standard_pixel_value_LR1)+", "+str(max_pixel_value_LR1)+"]\n→ "+str(reference_section_for_correction*100)+"(%)"
        ax4.text(x, max(list_max)*1.1*0.5, text, color='black', fontsize='12')
        ax6.text(x, max(list_max)*1.1*0.5, text, color='black', fontsize='12')

        # Draw rectangle
        rect = plt.Rectangle((_standard_pixel_value_LR1, 0), max_pixel_value_LR1-_standard_pixel_value_LR1, max(list_max)*1.1, fc='black', alpha=0.3)
        ax4.add_patch(rect)
        rect = plt.Rectangle((_standard_pixel_value_LR1, 0), max_pixel_value_LR1-_standard_pixel_value_LR1, max(list_max)*1.1, fc='black', alpha=0.3)
        ax6.add_patch(rect)



# Correct pixel value for each RGB
def correct_pixel_value(_rgb_img, _param):
    corrected_img_RGB = np.empty((_rgb_img.shape[0], _rgb_img.shape[1], 3), dtype=np.uint8)

    # Apply correction
    corrected_img_RGB[:, :, 0] = cv2.multiply(_rgb_img[:, :, 0], _param) # R
    corrected_img_RGB[:, :, 1] = cv2.multiply(_rgb_img[:, :, 1], _param) # G
    corrected_img_RGB[:, :, 2] = cv2.multiply(_rgb_img[:, :, 2], _param) # B

    return corrected_img_RGB



def preProcess():
    print("\n\n===== Pre-processing =====")
    print("Input image(RGB)\n>", img_in_RGB.shape) # （height, width, channel）

    # Calc all number of pixels of the input image
    N_all = img_in_RGB.shape[0]*img_in_RGB.shape[1]
    print("\nN_all\n>", N_all, "(pixels)")

    print("\n-----", args[1], "-----")
    # Calc number of pixels that pixel value is not 0
    img_in_Gray     = cv2.cvtColor(img_in_RGB, cv2.COLOR_RGB2GRAY)
    N_all_nonzero   = np.sum(img_in_Gray > 0)
    print("N_all_nonzero\n>", N_all_nonzero, "(pixels)")

    print("\n-----", args[2], "-----")
    # Calc max pixel value of the input image(LR=1)
    img_in_Gray_LR1     = cv2.cvtColor(img_in_RGB_LR1, cv2.COLOR_RGB2GRAY)
    N_all_nonzero_LR1   = np.sum(img_in_Gray_LR1 > 0)
    max_pixel_value_LR1 = np.max(img_in_Gray_LR1)
    print("Max pixel value\n>", max_pixel_value_LR1, "(pixel value)")

    # Calc the ratio of the maximum pixel value
    num_max_pixel_value_LR1 = np.sum(img_in_Gray_LR1 == max_pixel_value_LR1)
    print("\nNumber of max pixel value (", max_pixel_value_LR1, ")\n>", num_max_pixel_value_LR1, "(pixels)")
    ratio_max_pixel_value = num_max_pixel_value_LR1 / N_all_nonzero_LR1
    # ratio_max_pixel_value = round(ratio_max_pixel_value, 4)
    ratio_max_pixel_value = round(ratio_max_pixel_value, 8)
    print("\nRatio of the max pixel value\n>", ratio_max_pixel_value, " (", round(ratio_max_pixel_value*100, 2), "(%) )")

    # Calc most frequent pixel value
    img_in_Gray_nonzero_LR1         = img_in_Gray_LR1[img_in_Gray_LR1 > 0]
    bincount = np.bincount(img_in_Gray_nonzero_LR1)
    most_frequent_pixel_value_LR1   = np.argmax( bincount )
    print("\nMost frequent pixel value\n>", most_frequent_pixel_value_LR1, "(pixel value)")

    # To avoid noise
    if most_frequent_pixel_value_LR1 == 255:
        max_pixel_value_LR1 = 254
        print("\n** Changed max pixel value as follows.")
        print("** >", 255, " → ", 254)

    return N_all_nonzero, N_all_nonzero_LR1, img_in_Gray_LR1, max_pixel_value_LR1, ratio_max_pixel_value, most_frequent_pixel_value_LR1



def determineCorrectionParameter_UsingSubstituion255(_reference_section, _ratio_max_pixel_value):
    print("\n========================================================================================")
    print("** There is a possibility that pixel value \"255\" is too much in the input image(LR=1).")

    # Set reference section for searching substituion of 255
    print("\n** reference_section\n** >", _reference_section, "(", _reference_section*100, "(%) )")
            
    # Determine standard pixel value in the input image(LR=1)
    tmp_reference_section = 0.0
    standard_pixel_value_LR1 = 254
    while tmp_reference_section < _reference_section:
        # Temporarily, calc
        sum_pixels_in_section = np.sum( (standard_pixel_value_LR1 <= img_in_Gray_LR1) & (img_in_Gray_LR1 < 255) )
        tmp_reference_section = sum_pixels_in_section / N_all_nonzero_LR1

        # Next pixel value
        standard_pixel_value_LR1 -= 1 

    # print("\n** final reference section")
    # print("** >", tmp_reference_section*100, "(%)")

    if standard_pixel_value_LR1 < 0:
        standard_pixel_value_LR1 = 0
    print("\n** Standard pixel value")
    print("** >", standard_pixel_value_LR1, "(pixel value)")

    # Calc median pixel value in the section b/w standard pixel value and maximum pixel value(255)
    section_bw_standard_255_LR1 = img_in_Gray_LR1[ (standard_pixel_value_LR1 <= img_in_Gray_LR1) & (img_in_Gray_LR1 < 255) ]
    median_bw_standard_255_LR1  = int(np.median(section_bw_standard_255_LR1))
    print("\n** Median pixel value in the section between", standard_pixel_value_LR1, "and 255")
    print("** >", median_bw_standard_255_LR1, "(pixel value)")

    # Update ratio_max_pixel_value
    ratio_old = _ratio_max_pixel_value
    _ratio_max_pixel_value = np.sum(img_in_Gray_LR1 == median_bw_standard_255_LR1) / N_all_nonzero_LR1
    _ratio_max_pixel_value = round(_ratio_max_pixel_value, 4)
    print("\n** Ratio of the pixel value", median_bw_standard_255_LR1)
    print("** >", _ratio_max_pixel_value, "(", round(_ratio_max_pixel_value*100, 3), "(%) )")

    print("\n** Changed ratio as follows.")
    print("** >", ratio_old, " → ", _ratio_max_pixel_value)
    print("** >", round(ratio_old*100, 2), "(%) → ", round(_ratio_max_pixel_value*100, 3), "(%)")

    print("========================================================================================")

    # Determine parameter
    p = p_init
    tmp_ratio = 0.0
    while tmp_ratio < _ratio_max_pixel_value:
        # Temporarily, correct input image with p
        tmp_corrected_img_RGB   = correct_pixel_value(img_in_RGB, p)
        tmp_corrected_img_Gray  = cv2.cvtColor(tmp_corrected_img_RGB, cv2.COLOR_RGB2GRAY)

        # Then, calc ratio of max pixel value(LR=1)
        tmp_ratio = np.sum(max_pixel_value_LR1 <= tmp_corrected_img_Gray) / N_all_nonzero

        # Update parameter
        p += p_interval

    p_final = round(p, 2)

    return p_final, median_bw_standard_255_LR1, standard_pixel_value_LR1



def determineCorrectionParameter_UsingSection(_reference_section_for_correction, _most_frequent_pixel_value_LR1):
    print("\n====================================================================")
    print("** The max pixel value is less than \"255\" in the input image(LR=1).")

    # Set reference section for searching substituion of 255
    print("\n** reference_section_for_correction\n** >", _reference_section_for_correction, "(", _reference_section_for_correction*100, "(%) )")

    # Determine standard pixel value in the input image(LR=1)
    tmp_reference_section = 0.0
    standard_pixel_value_LR1 = max_pixel_value_LR1
    while tmp_reference_section < _reference_section_for_correction:
        # Temporarily, calc    
        if _most_frequent_pixel_value_LR1 == 255:
            sum_pixels_in_section = np.sum( (standard_pixel_value_LR1 <= img_in_Gray_LR1) & (img_in_Gray_LR1 <= 254) )

        else:
            sum_pixels_in_section = np.sum( (standard_pixel_value_LR1 <= img_in_Gray_LR1) )
        
        tmp_reference_section = sum_pixels_in_section / N_all_nonzero_LR1

        # Next pixel value
        standard_pixel_value_LR1 -= 1

    print("\n** Standard pixel value")
    print("** >", standard_pixel_value_LR1, "(pixel value)")

    print("\n** Reference section")
    print("** >", standard_pixel_value_LR1, "(pixel value) ~", max_pixel_value_LR1, "(pixel value)")
    print("** >", _reference_section_for_correction*100, "(%)")

    print("====================================================================")

    # Determine parameter
    p = p_init
    tmp_ratio = 0.0
    while tmp_ratio < _reference_section_for_correction:
        # Temporarily, correct input image with p
        tmp_corrected_img_RGB   = correct_pixel_value(img_in_RGB, p)
        tmp_corrected_img_Gray  = cv2.cvtColor(tmp_corrected_img_RGB, cv2.COLOR_RGB2GRAY)

        # Then, calc ratio of max pixel value(LR=1)
        tmp_sum_pixels_in_section = np.sum(standard_pixel_value_LR1 <= tmp_corrected_img_Gray)
        tmp_ratio = tmp_sum_pixels_in_section / N_all_nonzero

        # Update parameter
        p += p_interval

    p_final = round(p, 2)

    print("\n** tmp_ratio")
    print("** >", tmp_ratio*100, "(%)")

    return p_final, standard_pixel_value_LR1



def determineCorrectionParameter(_ratio_max_pixel_value):
    # Determine parameter
    p = p_init
    tmp_ratio = 0.0
    while tmp_ratio < _ratio_max_pixel_value:
        # Temporarily, correct input image with p
        tmp_corrected_img_RGB   = correct_pixel_value(img_in_RGB, p)
        tmp_corrected_img_Gray  = cv2.cvtColor(tmp_corrected_img_RGB, cv2.COLOR_RGB2GRAY)

        # Then, calc ratio of max pixel value(LR=1)
        tmp_ratio = np.sum(tmp_corrected_img_Gray == 255) / N_all_nonzero

        # Update parameter
        p += p_interval

    p_final = round(p, 2)

    return p_final



def correctPixelValue(_p_final, _standard_pixel_value_LR1=None, _median_bw_standard_255_LR1=None):
    # Create corrected image
    img_corrected_RGB  = correct_pixel_value(img_in_RGB, _p_final)
    img_corrected_Gray = cv2.cvtColor(img_corrected_RGB, cv2.COLOR_RGB2GRAY)

    print("\n\n===== Result =====")
    print("p_final\n>", _p_final)
    print("\nThe ratio at which pixel value finally reached 255\n>", round(np.sum(img_corrected_Gray==255) / N_all_nonzero * 100, 2), "(%)")
    print("\n")

    # Create figure
    if _median_bw_standard_255_LR1 is not None:
        plot_histogram(img_in_RGB_LR1, img_in_RGB, img_corrected_RGB, _standard_pixel_value_LR1, _median_bw_standard_255_LR1)

    elif _standard_pixel_value_LR1 is not None:
        plot_histogram(img_in_RGB_LR1, img_in_RGB, img_corrected_RGB, _standard_pixel_value_LR1)

    else:
        plot_histogram(img_in_RGB_LR1, img_in_RGB, img_corrected_RGB)

    return img_corrected_RGB



# Save figure and images
def saveFigureAndImages(_p_final, _img_in_RGB, _img_corrected_RGB):
    fig_name = "images/figure_"+str(_p_final)+".png"
    plt.savefig(fig_name)
    # plt.show()

    # convert color (RGB → BGR)
    img_in_BGR = cv2.cvtColor(_img_in_RGB, cv2.COLOR_RGB2BGR)
    img_out_BGR = cv2.cvtColor(_img_corrected_RGB, cv2.COLOR_RGB2BGR)
    input_img_name = "images/input.jpg"
    corrected_img_name = "images/corrected_"+str(_p_final)+".jpg"
    cv2.imwrite(input_img_name, img_in_BGR)
    cv2.imwrite(corrected_img_name, img_out_BGR)

    #execCommand(fig_name, input_img_name, corrected_img_name)



# Exec. command
def execCommand(_fig_name, _input_img_name, _corrected_img_name):
    # preview_command = ['code', _fig_name, _input_img_name, _corrected_img_name]
    preview_command = ['open', _fig_name, _input_img_name, _corrected_img_name]
    try:
	    res = subprocess.check_call(preview_command)

    except:
	    print("ERROR")



if __name__ == "__main__":
    # Read input image
    img_in_RGB      = read_img(args[1])
    img_in_RGB_LR1  = read_img(args[2])

    # Get max pixel value (LR=1)
    N_all_nonzero, N_all_nonzero_LR1, img_in_Gray_LR1, max_pixel_value_LR1, ratio_max_pixel_value, most_frequent_pixel_value_LR1 = preProcess()

    # Check whether the most frequent pixel value is 255 (LR=1)
    if most_frequent_pixel_value_LR1 == 256:
        reference_section = 0.1
        p_final, median_bw_standard_255_LR1, standard_pixel_value_LR1 = determineCorrectionParameter_UsingSubstituion255(reference_section, ratio_max_pixel_value)
        img_corrected_RGB = correctPixelValue(p_final, standard_pixel_value_LR1, median_bw_standard_255_LR1)

    # Check whether the max pixel value is less than 255 (LR=1)
    elif max_pixel_value_LR1 <= 255:
        reference_section_for_correction = 0.01 # 1%
        p_final, standard_pixel_value_LR1 = determineCorrectionParameter_UsingSection(reference_section_for_correction, most_frequent_pixel_value_LR1)
        img_corrected_RGB = correctPixelValue(p_final, standard_pixel_value_LR1)

    else: # If (max_pixel_value_LR1 == 255) & (most_frequent_pixel_value_LR1 != 255) :
        p_final = determineCorrectionParameter(ratio_max_pixel_value)
        img_corrected_RGB = correctPixelValue(p_final)

    # Save figure and images
    saveFigureAndImages(p_final, img_in_RGB, img_corrected_RGB)