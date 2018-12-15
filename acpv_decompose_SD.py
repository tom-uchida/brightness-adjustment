import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cycler
import matplotlib.gridspec as gridspec
import cv2
import subprocess
import sys
from scipy import stats

args = sys.argv
if len(args) != 3:
    raise Exception('\nUSAGE\n> $ python acpv_decompose_SD.py [input_image_data] [input_image_data(LR=1)]')
    raise Exception('\n\nFor example\n> $ python acpv_decompose_SD.py input_image.bmp input_image_LR1.bmp\n')
    sys.exit()

plt.style.use('seaborn-white')

colors = cycler('color', ['#EE6666', '#3388BB', '#9988DD', '#EECC55', '#88BB44', '#FFBBBB'])
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True, grid=False, prop_cycle=colors)
plt.rc('grid', color='w', linestyle='solid')
plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=2)

plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["mathtext.rm"] = "Times New Roman"



# ---------------------------------
# ----- Set initial parameter -----
# ---------------------------------
print("\n===== Initial parameter =====")
input_image_data        = args[1]
input_image_data_LR1    = args[2]
p_init                  = 1.0
p_interval              = 0.01
reference_section       = 0.1 # 10%
print("input_image_data\n>",            input_image_data, "(args[1])")
print("\ninput_image_data(LR=1)\n>",    input_image_data_LR1, "(args[2])")
print("\np_init\n>",                    p_init)
print("\np_interval\n>",                p_interval)
print("\nreference_section\n>",         reference_section*100, "(%)")



# Read input image
def readImage(_img_name):
    # read input image
    img_BGR = cv2.imread(_img_name)

    # convert color (BGR → RGB)
    img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

    return img_RGB



# RGB histogram
# def rgbHist(_img_rgb, _ax):    
#     R_nonzero = _img_rgb[:,:,0][_img_rgb[:,:,0] > 0]
#     G_nonzero = _img_rgb[:,:,1][_img_rgb[:,:,1] > 0]
#     B_nonzero = _img_rgb[:,:,2][_img_rgb[:,:,2] > 0]
#     _ax.hist(R_nonzero.ravel(), bins=50, color='r', alpha=0.5, label="R")
#     _ax.hist(G_nonzero.ravel(), bins=50, color='g', alpha=0.5, label="G")
#     _ax.hist(B_nonzero.ravel(), bins=50, color='b', alpha=0.5, label="B")
#     _ax.legend()

#     _ax.set_title('RGB histogram')
#     _ax.set_xlim([-5, 260])
    
#     return _ax



# Grayscale histogram
def grayscaleHist(_img_Gray, _ax, _title):    
    img_Gray_nonzero = _img_Gray[_img_Gray > 0]
    _ax.hist(img_Gray_nonzero.ravel(), bins=50, color='black')

    _ax.set_title('Histogram ('+_title+')')
    _ax.set_xlim([-5, 260])
    
    return _ax



# Histograms of Input(LR=1), Input and Corrected
def comparativeHist(_img_in_rgb_LR1, _img_in_rgb, _img_out_rgb, _ax, _y_max):
    # Convert RGB to Grayscale
    img_in_Gray_LR1         = cv2.cvtColor(_img_in_rgb_LR1, cv2.COLOR_RGB2GRAY)
    img_in_Gray_LR1_nonzero = img_in_Gray_LR1[img_in_Gray_LR1 > 0]
    img_in_Gray             = cv2.cvtColor(_img_in_rgb, cv2.COLOR_RGB2GRAY)
    img_in_Gray_nonzero     = img_in_Gray[img_in_Gray > 0]
    img_out_Gray            = cv2.cvtColor(_img_out_rgb, cv2.COLOR_RGB2GRAY)
    img_out_Gray_nonzero    = img_out_Gray[img_out_Gray > 0]
    
    # Input image(LR=1)
    mean_in_LR1 = int(np.mean(img_in_Gray_LR1_nonzero))
    _ax.hist(img_in_Gray_LR1_nonzero.ravel(), bins=50, alpha=0.5, label="Input image ($L_{\mathrm{R}}=1$)", color='#1F77B4')
    _ax.axvline(mean_in_LR1, color='#1F77B4')
    _ax.text(mean_in_LR1+5, _y_max*0.8, "mean:"+str(mean_in_LR1), color='#1F77B4', fontsize='12')

    # Input image
    mean_in = int(np.mean(img_in_Gray_nonzero))
    _ax.hist(img_in_Gray_nonzero.ravel(), bins=50, alpha=0.5, label="Input image", color='#FF7E0F')
    _ax.axvline(mean_in, color='#FF7E0F')
    _ax.text(mean_in+5, _y_max*0.6, "mean:"+str(mean_in), color='#FF7E0F', fontsize='12')

    # Corrected image
    mean_out = int(np.mean(img_out_Gray_nonzero))
    _ax.hist(img_out_Gray_nonzero.ravel(), bins=50, alpha=0.5, label="Corrected image", color='#2C9F2C')
    _ax.axvline(mean_out, color='#2C9F2C')
    _ax.text(mean_out+5, _y_max*0.7, "mean:"+str(mean_out), color='#2C9F2C', fontsize='12')

    _ax.set_title('Comparative histogram')
    _ax.set_xlabel("Pixel value")
    _ax.set_ylabel("Number of pixels")
    _ax.legend(fontsize='12')
    
    return _ax



# Create histogram (Low, High)
def plotHist4LowAndHighImage(_p_final, _img_in_RGB, _img_out_RGB, _title, _ylim):
    # Convert RGB to Grayscale
    img_in_Gray = cv2.cvtColor(_img_in_RGB, cv2.COLOR_RGB2GRAY)
    img_out_Gray = cv2.cvtColor(_img_out_RGB, cv2.COLOR_RGB2GRAY)

    fig = plt.figure(figsize=(10,7))
    gs = gridspec.GridSpec(2,2)

    # Input image
    ax1 = fig.add_subplot(gs[0,0])
    ax1.set_title('Input image'+' ('+_title+')')
    ax1.imshow(_img_in_RGB)
    ax1.set_xticks([]), ax1.set_yticks([]) # off scale

    # Corrected image
    ax2 = fig.add_subplot(gs[0,1])
    ax2.set_title('Corrected image'+' ('+_title+')')
    ax2.imshow(_img_out_RGB)
    ax2.set_xticks([]), ax2.set_yticks([])

    # Grayscale histogram(Input image)
    ax3 = fig.add_subplot(gs[1,0])
    ax3 = grayscaleHist(img_in_Gray, ax3, "Input image")

    # Grayscale histogram(Corrected image)
    ax4 = fig.add_subplot(gs[1,1])
    ax4 = grayscaleHist(img_out_Gray, ax4, "Corrected image")

    # Unify ylim b/w input image and corrected image
    ax3.set_ylim([0, _ylim])
    ax4.set_ylim([0, _ylim])

    # Draw line
    ax3.axvline(boundary_pixel_value, color='red')
    ax3.text(boundary_pixel_value+5, _ylim*0.7, "boundary:"+str(boundary_pixel_value), color='red', fontsize='12')
    ax4.axvline(boundary_pixel_value, color='red')
    ax4.text(boundary_pixel_value+5, _ylim*0.7, "boundary:"+str(boundary_pixel_value), color='red', fontsize='12')

    # Save figure
    fig_name = "images/figure_"+str(_p_final)+"_"+str(_title)+".png"
    plt.savefig(fig_name)



# Create histogram (Input(LR1), Input, Corrected)
def plotHist4LR1AndInAndOut(_p_final, _img_in_RGB_LR1, _img_in_RGB, _img_out_RGB):
    # Convert RGB to Grayscale
    img_in_Gray_LR1 = cv2.cvtColor(_img_in_RGB_LR1, cv2.COLOR_RGB2GRAY)
    img_in_Gray     = cv2.cvtColor(_img_in_RGB, cv2.COLOR_RGB2GRAY)
    img_out_Gray    = cv2.cvtColor(_img_out_RGB, cv2.COLOR_RGB2GRAY)

    fig = plt.figure(figsize=(12,10)) # (wdith, height)
    gs = gridspec.GridSpec(3,3)

    # Input image(LR=1)
    ax1 = fig.add_subplot(gs[0,0])
    ax1.set_title('Input image ($L_{\mathrm{R}}=1$)')
    ax1.imshow(_img_in_RGB_LR1)
    ax1.set_xticks([]), ax1.set_yticks([])

    # Input image
    ax2 = fig.add_subplot(gs[0,1])
    ax2.set_title('Input image')
    ax2.imshow(_img_in_RGB)
    ax2.set_xticks([]), ax2.set_yticks([]) # off scale

    # Corrected image
    ax3 = fig.add_subplot(gs[0,2])
    ax3.set_title('Corrected image')
    ax3.imshow(_img_out_RGB)
    ax3.set_xticks([]), ax3.set_yticks([])

    # Grayscale histogram(Input image(LR=1))
    ax4 = fig.add_subplot(gs[1,0])
    ax4 = grayscaleHist(img_in_Gray_LR1, ax4, "Input image ($L_{\mathrm{R}}=1$)")

    # Grayscale histogram(Input image)
    ax5 = fig.add_subplot(gs[1,1])
    ax5 = grayscaleHist(img_in_Gray, ax5, "Input image")

    # Grayscale histogram(Corrected image)
    ax6 = fig.add_subplot(gs[1,2])
    ax6 = grayscaleHist(img_out_Gray, ax6, "Corrected image")

    # Unify ylim b/w input image and corrected image
    hist_in_LR1, bins_in_LR1    = np.histogram(img_in_Gray_LR1[img_in_Gray_LR1>0],  50)
    hist_in, bins_in            = np.histogram(img_in_Gray[img_in_Gray>0],          50)
    hist_out, bins_out          = np.histogram(img_out_Gray[img_out_Gray>0],        50)
    list_gray_max = [max(hist_in_LR1), max(hist_in), max(hist_out)]
    ax4.set_ylim([0, max(list_gray_max)*1.1])
    ax5.set_ylim([0, max(list_gray_max)*1.1])
    ax6.set_ylim([0, max(list_gray_max)*1.1])
    
    # Comparative histogram (Input(LR1), Input, Corrected)
    ax7 = fig.add_subplot(gs[2,:])
    ax7 = comparativeHist(_img_in_RGB_LR1, _img_in_RGB, _img_out_RGB, ax7, max(list_gray_max)*1.1)
    ax7.set_ylim([0, max(list_gray_max)*1.1])

    # Draw line
    ax5.axvline(boundary_pixel_value, color='red')
    ax5.text(boundary_pixel_value+5, max(list_gray_max)*1.1*0.7, "boundary:"+str(boundary_pixel_value), color='red', fontsize='12')
    ax6.axvline(boundary_pixel_value, color='red')
    ax6.text(boundary_pixel_value+5, max(list_gray_max)*1.1*0.7, "boundary:"+str(boundary_pixel_value), color='red', fontsize='12')

    # Save figure
    fig_name = "images/figure_"+str(_p_final)+".png"
    plt.savefig(fig_name)

    # return ylim
    return max(list_gray_max)*1.1



def preProcess():
    print("\n\n===== Pre-processing =====")
    print("Input image(RGB)\n>", img_in_RGB.shape)

    # Calc all number of pixels of the input image
    N_all = img_in_RGB.shape[0]*img_in_RGB.shape[1]
    print("\nN_all\n>", N_all, "(pixels)")

    # ------------------------
    # ------ Input image -----
    # ------------------------
    print("\n-----", args[1], "-----")
    # Calc number of pixels that pixel value is not 0
    N_all_nonzero = np.sum(img_in_Gray > 0)
    print("N_all_nonzero\n>", N_all_nonzero, "(pixels)")

    # Calc mean pixel value of the input image
    img_in_Gray_nonzero = img_in_Gray[img_in_Gray > 0]
    mean = int(img_in_Gray_nonzero.mean())
    print("\nMean pixel value\n>", mean, "(pixel value)")

    # Calc Standard Deviation pixel value of the input image
    sd = int(img_in_Gray_nonzero.std())
    print("\nStandard Deviation\n>", sd, "(pixel value)")

    # -------------------------------
    # ------ Input image (LR=1) -----
    # -------------------------------
    print("\n-----", args[2], "-----")
    N_all_nonzero_LR1   = np.sum(img_in_Gray_LR1 > 0)
    print("N_all_nonzero_LR1\n>", N_all_nonzero_LR1, "(pixels)")

    # Calc mean pixel value of the input image(LR=1)
    img_in_Gray_nonzero_LR1 = img_in_Gray_LR1[img_in_Gray_LR1 > 0]
    mean_LR1 = int(img_in_Gray_nonzero_LR1.mean())
    print("\nMean pixel value\n>", mean_LR1, "(pixel value)")

    # Calc Standard Deviation pixel value of the input image(LR=1)
    sd_LR1 = int(img_in_Gray_nonzero_LR1.std())
    print("\nStandard Deviation\n>", sd_LR1, "(pixel value)")

    # Calc max pixel value of the input image(LR=1)
    max_pixel_value_LR1 = np.max(img_in_Gray_LR1)
    print("\nMax pixel value\n>", max_pixel_value_LR1, "(pixel value)")

    # Calc the ratio of the maximum pixel value
    num_max_pixel_value_LR1 = np.sum(img_in_Gray_LR1 == max_pixel_value_LR1)
    print("\nNumber of max pixel value (", max_pixel_value_LR1, ")\n>", num_max_pixel_value_LR1, "(pixels)")
    ratio_max_pixel_value = num_max_pixel_value_LR1 / N_all_nonzero_LR1
    # ratio_max_pixel_value = round(ratio_max_pixel_value, 4)
    ratio_max_pixel_value = round(ratio_max_pixel_value, 8)
    print("\nRatio of the max pixel value\n>", ratio_max_pixel_value, " (", round(ratio_max_pixel_value*100, 2), "(%) )")

    # Check whether the maximum pixel value is 255 in the input image(LR=1)
    if max_pixel_value_LR1 == 255:
        # Calc most frequent pixel value
        img_in_Gray_nonzero_LR1         = img_in_Gray_LR1[img_in_Gray_LR1 > 0]
        bincount = np.bincount(img_in_Gray_nonzero_LR1)
        most_frequent_pixel_value_LR1   = np.argmax( bincount )
        print("\nMost frequent pixel value\n>", most_frequent_pixel_value_LR1, "(pixel value)")

        # Check whether the most frequent pixel value is 255 in the input image(LR=1)
        if most_frequent_pixel_value_LR1 == 255:
            print("\n========================================================================================")
            print("** There is a possibility that pixel value \"255\" is too much in the input image(LR=1).")
            
            # Determine standard pixel value in the input image(LR=1)
            tmp_reference_section = 0.0
            standard_pixel_value_LR1 = 254
            while tmp_reference_section < reference_section:
                # Temporarily, calc
                sum_pixels_in_section = np.sum( (standard_pixel_value_LR1 <= img_in_Gray_LR1) & (img_in_Gray_LR1 < 255) )
                tmp_reference_section = sum_pixels_in_section / N_all_nonzero_LR1

                # Next pixel value
                standard_pixel_value_LR1 -= 1

            # print("\n** final reference section")
            # print("** >", tmp_reference_section*100, "(%)")

            print("\n** Standard pixel value")
            print("** >", standard_pixel_value_LR1, "(pixel value)")

            # Calc median pixel value in the section b/w standard pixel value and maximum pixel value(255)
            section_bw_standard_255_LR1 = img_in_Gray_LR1[ (standard_pixel_value_LR1 <= img_in_Gray_LR1) & (img_in_Gray_LR1 < 255) ]
            median_bw_standard_255_LR1  = int(np.median(section_bw_standard_255_LR1))
            print("\n** Median pixel value in the section between", standard_pixel_value_LR1, "and 255")
            print("** >", median_bw_standard_255_LR1, "(pixel value)")

            # Update ratio_max_pixel_value
            ratio_old = ratio_max_pixel_value
            ratio_max_pixel_value = np.sum(img_in_Gray_LR1 == median_bw_standard_255_LR1) / N_all_nonzero_LR1
            ratio_max_pixel_value = round(ratio_max_pixel_value, 4)
            print("\n** Ratio of the pixel value", median_bw_standard_255_LR1)
            print("** >", ratio_max_pixel_value, "(", round(ratio_max_pixel_value*100, 3), "(%) )")

            print("\n** Changed ratio as follows.")
            print("** >", ratio_old, " → ", ratio_max_pixel_value)
            print("** >", round(ratio_old*100, 2), "(%) → ", round(ratio_max_pixel_value*100, 3), "(%)")

            print("========================================================================================")

    return N_all_nonzero, N_all_nonzero_LR1, max_pixel_value_LR1, ratio_max_pixel_value, mean, mean_LR1, sd, sd_LR1



# Decompose the input image into two images
def decomposeImage(_boundary_pixel_value):
    print("\n** Decomposition boundary pixel value\n** >", _boundary_pixel_value, "(pixel value)")
    low_num_nonzero  = np.count_nonzero( (img_in_Gray <= _boundary_pixel_value) & (img_in_Gray > 0) )
    low_num  = np.count_nonzero(  img_in_Gray <= _boundary_pixel_value )
    high_num = np.count_nonzero(  img_in_Gray >  _boundary_pixel_value )
    print("\nNumber of \"low\" pixel values\n>", low_num_nonzero, "(pixels) out of", N_all_nonzero, "(pixels)")
    print(">", round(low_num_nonzero/N_all_nonzero*100,1), "(%)")
    print("\nNumber of \"high\" pixel values\n>", high_num, "(pixels) out of", N_all_nonzero, "(pixels)")
    print(">", round(high_num/N_all_nonzero*100,1), "(%)")

    # ndarray dtype:bool
    low_index_bool  =  img_in_Gray <= _boundary_pixel_value
    high_index_bool =  ~low_index_bool

    # Separate the input image into R,G,B channel
    img_in_R, img_in_G, img_in_B = img_in_RGB[:,:,0], img_in_RGB[:,:,1], img_in_RGB[:,:,2]

    # Apply decomposition
    low_R  = np.where(low_index_bool,  img_in_R, 0)
    low_G  = np.where(low_index_bool,  img_in_G, 0)
    low_B  = np.where(low_index_bool,  img_in_B, 0)
    high_R = np.where(high_index_bool, img_in_R, 0)
    high_G = np.where(high_index_bool, img_in_G, 0)
    high_B = np.where(high_index_bool, img_in_B, 0)

    # Create low and high pixel value image
    low_img_in_RGB, high_img_in_RGB = img_in_RGB.copy(), img_in_RGB.copy()
    low_img_in_RGB[:,:,0],  low_img_in_RGB[:,:,1],  low_img_in_RGB[:,:,2]  = low_R,  low_G,  low_B
    high_img_in_RGB[:,:,0], high_img_in_RGB[:,:,1], high_img_in_RGB[:,:,2] = high_R, high_G, high_B

    return low_img_in_RGB, high_img_in_RGB



# Correct pixel value for each RGB
def correctPixelValue(_rgb_img, _param):
    # Apply change
    corrected_img_RGB = np.empty((_rgb_img.shape[0], _rgb_img.shape[1], 3), dtype=np.uint8)
    corrected_img_RGB[:, :, 0] = cv2.multiply(_rgb_img[:, :, 0], _param) # R
    corrected_img_RGB[:, :, 1] = cv2.multiply(_rgb_img[:, :, 1], _param) # G
    corrected_img_RGB[:, :, 2] = cv2.multiply(_rgb_img[:, :, 2], _param) # B

    return corrected_img_RGB



# Calc parameter for low pixel value image
def determineParameter4LowPixelValueImage(_target_pixel_value):
    p = p_init
    tmp_ratio = 0.0
    ratio_target_pixel_value = np.sum(img_in_Gray_LR1 == _target_pixel_value) / N_all_nonzero_LR1
    while tmp_ratio < ratio_target_pixel_value:
        tmp_img_RGB     = correctPixelValue(low_img_in_RGB, p)
        tmp_img_Gray    = cv2.cvtColor(tmp_img_RGB, cv2.COLOR_RGB2GRAY)

        # Temporarily, calc ratio of the target pixel value
        tmp_ratio = np.sum(tmp_img_Gray == _target_pixel_value) / N_all_nonzero
        
        # Update parameter
        p += p_interval

    # Result
    p_final_low = round(p, 2)
    print("\n\n===== Result for \"low pixel value image\" =====")
    print("ratio_target_pixel_value\n>", ratio_target_pixel_value)
    print("\ntarget_pixel_value\n>", _target_pixel_value)
    print("\np_final_low\n>", p_final_low)

    # Make low output image
    low_img_out_RGB = correctPixelValue(low_img_in_RGB, p_final_low)

    return low_img_out_RGB, p_final_low



# Calc parameter for high pixel value image
def determineParameter4HighPixelValueImage():
    p = p_init
    tmp_ratio = 0.0
    while tmp_ratio < ratio_max_pixel_value_LR1:
        tmp_img_RGB     = correctPixelValue(high_img_in_RGB, p)
        tmp_img_Gray    = cv2.cvtColor(tmp_img_RGB, cv2.COLOR_RGB2GRAY)

        # Temporarily, calc ratio of the max pixel value(LR=1)
        tmp_ratio = np.sum(tmp_img_Gray == max_pixel_value_LR1) / N_all_nonzero
        
        # Update parameter
        p += p_interval

    # Result
    p_final_high = round(p, 2)
    print("\n===== Result for \"high pixel value image\" =====")
    print("p_final_high\n>", p_final_high)

    # Make high output image
    high_img_out_RGB = correctPixelValue(high_img_in_RGB, p_final_high)

    return high_img_out_RGB, p_final_high



# Synthesize low and high pixel value images
def synthesizeLowAndHighPixelValueImages():
    img_out_RGB  = cv2.scaleAdd(low_img_out_RGB, 1.0, high_img_out_RGB)
    img_out_Gray = cv2.cvtColor(img_out_RGB, cv2.COLOR_RGB2GRAY)

    # Calc finally 
    num_max_pixel_value_out = np.sum(img_out_Gray == max_pixel_value_LR1)
    print("\n** The ratio at which pixel value finally reached", max_pixel_value_LR1, "\n** >", round(num_max_pixel_value_out / N_all_nonzero * 100, 2), "(%)\n")

    return img_out_RGB



# Save images
def saveCorrectedImages():
    #plt.show()

    # Convert color (RGB → BGR)
    img_in_BGR              = cv2.cvtColor(img_in_RGB, cv2.COLOR_RGB2BGR)
    img_out_BGR_low         = cv2.cvtColor(low_img_out_RGB, cv2.COLOR_RGB2BGR)
    img_out_BGR_high        = cv2.cvtColor(high_img_out_RGB, cv2.COLOR_RGB2BGR)
    img_out_BGR             = cv2.cvtColor(img_out_RGB, cv2.COLOR_RGB2BGR)
    input_img_name          = "images/input.jpg"
    low_output_img_name     = "images/low_corrected_"+str(p_final_low)+".jpg"
    high_output_img_name    = "images/high_corrected_"+str(p_final_high)+".jpg"
    output_img_name         = "images/corrected_low-"+str(p_final_low)+"_high-"+str(p_final_high)+".jpg"
    cv2.imwrite(input_img_name, img_in_BGR)
    cv2.imwrite(low_output_img_name, img_out_BGR_low)
    cv2.imwrite(high_output_img_name, img_out_BGR_high)
    cv2.imwrite(output_img_name, img_out_BGR)

    # Save low and high images
    low_img_in_BGR  = cv2.cvtColor(low_img_in_RGB, cv2.COLOR_RGB2BGR)
    high_img_in_BGR = cv2.cvtColor(high_img_in_RGB, cv2.COLOR_RGB2BGR)
    low_img_name    = "images/low.jpg"
    high_img_name   = "images/high.jpg"
    cv2.imwrite(low_img_name, low_img_in_BGR)
    cv2.imwrite(high_img_name, high_img_in_BGR)



if __name__ == "__main__":
    # Read input images
    img_in_RGB      = readImage(args[1])
    img_in_RGB_LR1  = readImage(args[2])

    # Convert RGB image to Grayscale image
    img_in_Gray     = cv2.cvtColor(img_in_RGB,      cv2.COLOR_RGB2GRAY)
    img_in_Gray_LR1 = cv2.cvtColor(img_in_RGB_LR1,  cv2.COLOR_RGB2GRAY)

    # Pre-process
    N_all_nonzero, N_all_nonzero_LR1, max_pixel_value_LR1, ratio_max_pixel_value_LR1, mean, mean_LR1, sd, sd_LR1 = preProcess()

    # Decompose input image
    boundary_pixel_value = mean+sd*2
    low_img_in_RGB, high_img_in_RGB = decomposeImage(boundary_pixel_value)

    # Correct low and high pixel value images
    low_img_out_RGB, p_final_low   = determineParameter4LowPixelValueImage(mean_LR1+sd_LR1*2)
    high_img_out_RGB, p_final_high = determineParameter4HighPixelValueImage()
    
    # Synthesize low and high pixel value images 
    img_out_RGB = synthesizeLowAndHighPixelValueImages()

    # Create figure with p_final
    ylim = plotHist4LR1AndInAndOut("corrected", img_in_RGB_LR1, img_in_RGB, img_out_RGB)
    plotHist4LowAndHighImage(p_final_low,  low_img_in_RGB,     low_img_out_RGB,    "Low",   ylim)
    plotHist4LowAndHighImage(p_final_high, high_img_in_RGB,    high_img_out_RGB,   "High",  ylim)

    

    # Save corrected images
    saveCorrectedImages()
