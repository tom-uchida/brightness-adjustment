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



# Set initial parameter
p_init = 1.0
p_interval = 0.01
reference_section = 0.1 # 10%
print("\n===== Initial parameter =====")
print("input_image_data\n>",            args[1], "(args[1])")
print("\ninput_image_data(LR=1)\n>",    args[2], "(args[2])")
print("\np_init\n>",                    p_init)
print("\np_interval\n>",                p_interval)
print("\nreference_section\n>",         reference_section, "(", reference_section*100, "(%) )")



# RGB histogram
def rgb_hist(_img_rgb, _ax):
    R_nonzero = _img_rgb[:,:,0][_img_rgb[:,:,0] > 0]
    G_nonzero = _img_rgb[:,:,1][_img_rgb[:,:,1] > 0]
    B_nonzero = _img_rgb[:,:,2][_img_rgb[:,:,2] > 0]
    _ax.hist(R_nonzero.ravel(), bins=50, color='r', alpha=0.5, label="R")
    _ax.hist(G_nonzero.ravel(), bins=50, color='g', alpha=0.5, label="G")
    _ax.hist(B_nonzero.ravel(), bins=50, color='b', alpha=0.5, label="B")
    _ax.legend()

    _ax.set_title('RGB histogram')
    
    return _ax



# Grayscale histogram
def grayscale_hist(_img_rgb, _ax):
    img_Gray = cv2.cvtColor(_img_rgb, cv2.COLOR_RGB2GRAY)
    img_Gray_nonzero = img_Gray[img_Gray > 0]
    _ax.hist(img_Gray_nonzero.ravel(), bins=50, color='black', alpha=1.0)

    _ax.set_title('Grayscale histogram')
    #_ax.set_xlim([-5,260])
    
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
    _ax.hist(img_in_Gray_LR1_nonzero.ravel(), bins=50, alpha=0.6, label="Input image (LR=1)", color='#1F77B4')
    _ax.axvline(mean_in_LR1, color='#1F77B4')
    _ax.text(mean_in_LR1+5, _y_max*0.8, "mean:"+str(mean_in_LR1), color='#1F77B4', fontsize='12')

    # input image
    mean_in = int(np.mean(img_in_Gray_nonzero))
    _ax.hist(img_in_Gray_nonzero.ravel(), bins=50, alpha=0.6, label="Input image", color='#FF7E0F')
    _ax.axvline(mean_in, color='#FF7E0F')
    _ax.text(mean_in+5, _y_max*0.6, "mean:"+str(mean_in), color='#FF7E0F', fontsize='12')

    # corrected image
    mean_out = int(np.mean(img_out_Gray_nonzero))
    _ax.hist(img_out_Gray_nonzero.ravel(), bins=50, alpha=0.6, label="Corrected image", color='#2C9F2C')
    _ax.axvline(mean_out, color='#2C9F2C')
    _ax.text(mean_out+5, _y_max*0.7, "mean:"+str(mean_out), color='#2C9F2C', fontsize='12')

    _ax.set_title('Comparative grayscale histograms')
    _ax.legend()
    
    return _ax



def plot_histogram(_img_in_RGB_LR1,  _img_in_RGB, _img_out_RGB, _median_bw_standard_255_LR1=None, _standard_pixel_value_LR1=None):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(3,3)
    x = np.arange(256)

    # Input image(LR=1)
    ax1 = fig.add_subplot(gs[0,0])
    ax1.set_title('Input image (LR=1)')
    ax1.imshow(_img_in_RGB_LR1)
    ax1.set_xticks([]), ax1.set_yticks([])

    # Input image
    ax2 = fig.add_subplot(gs[0,1])
    ax2.set_title('Input image')
    ax2.imshow(_img_in_RGB)
    ax2.set_xticks([]), ax2.set_yticks([])

    # Output image
    ax3 = fig.add_subplot(gs[0,2])
    ax3.set_title('Corrected image')
    ax3.imshow(_img_out_RGB)
    ax3.set_xticks([]), ax3.set_yticks([])

    # Histogram(input image(LR=1))
    ax4 = fig.add_subplot(gs[1,0])
    ax4 = grayscale_hist(_img_in_RGB_LR1, ax4)
    
    # Histogram(input image)
    ax5 = fig.add_subplot(gs[1,1])
    ax5 = rgb_hist(_img_in_RGB, ax5)

    # Histogram(output image)
    ax6 = fig.add_subplot(gs[1,2])
    ax6 = rgb_hist(_img_out_RGB, ax6)

    # Unify ylim b/w input image and corrected image
    hist_in_LR1, bins_in_LR1 = np.histogram(_img_in_RGB_LR1[_img_in_RGB_LR1>0], 50)
    hist_in,     bins_in     = np.histogram(_img_in_RGB[_img_in_RGB>0],         50)
    hist_out,    bins_out    = np.histogram(_img_out_RGB[_img_out_RGB>0],       50)
    list_max = [max(hist_in_LR1), max(hist_in), max(hist_out)]
    ax4.set_ylim([0, max(list_max)/2.5])
    ax5.set_ylim([0, max(list_max)/2.5])
    ax6.set_ylim([0, max(list_max)/2.5])

    # Histograms(Input(LR1), Input, Corrected)
    ax7 = fig.add_subplot(gs[2,:])
    ax7 = comparative_hist(_img_in_RGB_LR1, _img_in_RGB, _img_out_RGB, ax7, max(list_max)/2.5)
    ax7.set_ylim([0, max(list_max)/2.5])

    if _median_bw_standard_255_LR1 is not None:
        # Draw line
        ax4.axvline(_median_bw_standard_255_LR1, color='black', alpha=0.5)
        ax4.text(_median_bw_standard_255_LR1-100, max(list_max)/2.5*0.7, "median:"+str(_median_bw_standard_255_LR1), color='black')

        # Draw rectangle
        rect = plt.Rectangle((_standard_pixel_value_LR1, 0), 254-_standard_pixel_value_LR1, max(list_max)/2.5, fc='black', alpha=0.3)
        ax4.add_patch(rect)



# Read Input Image
def read_img(_img_name):
    # read input image
    img_BGR = cv2.imread(_img_name)

    # convert color (BGR → RGB)
    img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

    return img_RGB



# Correct pixel value for each RGB
def correct_pixel_value(_rgb_img, _param):
    corrected_img_RGB = np.empty((_rgb_img.shape[0], _rgb_img.shape[1], 3), dtype=np.uint8)

    # Apply correction
    corrected_img_RGB[:, :, 0] = cv2.multiply(_rgb_img[:, :, 0], _param) # R
    corrected_img_RGB[:, :, 1] = cv2.multiply(_rgb_img[:, :, 1], _param) # G
    corrected_img_RGB[:, :, 2] = cv2.multiply(_rgb_img[:, :, 2], _param) # B

    return corrected_img_RGB



def run():
    # Read input image
    img_in_RGB      = read_img(args[1])
    img_in_RGB_LR1  = read_img(args[2])

    print("\n\n===== Pre-processing =====")
    print("Input image(RGB)\n>", img_in_RGB.shape) # （height, width, channel）

    # Calc all number of pixels of the input image
    N_all = img_in_RGB.shape[0]*img_in_RGB.shape[1]
    print("\n-----", args[1], "-----")
    print("N_all\n>", N_all, "(pixels)")

    # Then, calc number of pixels that pixel value is not 0
    img_in_Gray     = cv2.cvtColor(img_in_RGB, cv2.COLOR_RGB2GRAY)
    N_all_nonzero   = np.sum(img_in_Gray > 0)
    print("\nN_all_nonzero\n>", N_all_nonzero, "(pixels)")

    # Calc max pixel value of the input image(LR=1)
    img_in_Gray_LR1     = cv2.cvtColor(img_in_RGB_LR1, cv2.COLOR_RGB2GRAY)
    N_all_nonzero_LR1   = np.sum(img_in_Gray_LR1 > 0)
    max_pixel_value_LR1 = np.max(img_in_Gray_LR1)
    print("\n-----", args[2], "-----")
    print("Max pixel value\n>", max_pixel_value_LR1, "(pixel value)")

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

    # Determine parameter
    p = p_init
    tmp_ratio_255 = 0.0
    # ratio_max_pixel_value = 0.01
    while tmp_ratio_255 < ratio_max_pixel_value:
        # Temporarily, correct input image with p
        tmp_corrected_img_RGB   = correct_pixel_value(img_in_RGB, p)
        tmp_corrected_img_Gray  = cv2.cvtColor(tmp_corrected_img_RGB, cv2.COLOR_RGB2GRAY)

        # Then, calc ratio of pixel value 255
        tmp_ratio_255 = np.sum(tmp_corrected_img_Gray == 255) / N_all_nonzero

        # Update parameter
        p += p_interval

    p_final = round(p, 2)

    # Create output image
    img_out_RGB  = correct_pixel_value(img_in_RGB, p_final)
    img_out_Gray = cv2.cvtColor(img_out_RGB, cv2.COLOR_RGB2GRAY)

    print("\n\n===== Result =====")
    print("p_final\n>", p_final)
    num_max_pixel_value_out = np.sum(img_out_Gray == max_pixel_value_LR1)
    print("\nNumber of max pixel value (", max_pixel_value_LR1, ")\n>", num_max_pixel_value_out, "(pixels)")
    print("\nThe ratio at which pixel value finally reached", max_pixel_value_LR1, "\n>", round(num_max_pixel_value_out / N_all_nonzero * 100, 2), "(%)")
    print("\n")

    # Create figure
    try:
        # Check if variable is defined
        median_bw_standard_255_LR1, standard_pixel_value_LR1

        plot_histogram(img_in_RGB_LR1, img_in_RGB, img_out_RGB, median_bw_standard_255_LR1, standard_pixel_value_LR1)

    except:
        plot_histogram(img_in_RGB_LR1, img_in_RGB, img_out_RGB)

    # Save figure and images
    save_figure_images(p_final, img_in_RGB, img_out_RGB)



# Save figure and images
def save_figure_images(_p_final, _img_in_RGB, _img_out_RGB):
    fig_name = "images/figure_"+str(_p_final)+".png"
    plt.savefig(fig_name)
    # plt.show()

    # convert color (RGB → BGR)
    img_in_BGR = cv2.cvtColor(_img_in_RGB, cv2.COLOR_RGB2BGR)
    img_out_BGR = cv2.cvtColor(_img_out_RGB, cv2.COLOR_RGB2BGR)
    input_img_name = "images/input.jpg"
    output_img_name = "images/corrected_"+str(_p_final)+".jpg"
    cv2.imwrite(input_img_name, img_in_BGR)
    cv2.imwrite(output_img_name, img_out_BGR)

    #exec_command(fig_name, input_img_name, output_img_name)



# Exec. command
def exec_command(_fig_name, _input_img_name, _output_img_name):
    preview_command = ['code', _fig_name, _input_img_name, _output_img_name]
    # preview_command = ['open', fig_name, input_img_name, output_img_name]
    try:
	    res = subprocess.check_call(preview_command)

    except:
	    print("ERROR")



if __name__ == "__main__":
    #set_initial_parameter()
    run()