# ヒストグラム平坦化
# http://lang.sist.chukyo-u.ac.jp/classes/OpenCV/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html

# Example: 
# python3 gamma_correct_test.py -f input_img.png -g 0.8

import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cycler
import matplotlib.gridspec as gridspec

plt.style.use('bmh')
colors = cycler('color', ['#EE6666', '#3388BB', '#9988DD', '#EECC55', '#88BB44', '#FFBBBB'])
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True, grid=False, prop_cycle=colors)

bgcolor = 0 # Background color: Black(0, 0, 0)

def read_image(_img_name):
    img_BGR = cv2.imread(_img_name)
    img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

    return img_RGB

def create_RGB_histogram(_img_rgb, _ax, _title):
    R_nonzero = _img_rgb[:,:,0][_img_rgb[:,:,0] != bgcolor]
    G_nonzero = _img_rgb[:,:,1][_img_rgb[:,:,1] != bgcolor]
    B_nonzero = _img_rgb[:,:,2][_img_rgb[:,:,2] != bgcolor]
    _ax.hist(R_nonzero.ravel(), bins=50, color='r', alpha=0.5, label="R")
    _ax.hist(G_nonzero.ravel(), bins=50, color='g', alpha=0.5, label="G")
    _ax.hist(B_nonzero.ravel(), bins=50, color='b', alpha=0.5, label="B")

    _ax.set_title(_title)
    _ax.set_xlim([-5,260])
    
    return _ax

def create_figure(_img_in_RGB, _img_in_RGB_equ):
    fig = plt.figure(figsize=(10, 8)) # figsize=(width, height)
    gs  = gridspec.GridSpec(2,2)

    ax1 = fig.add_subplot(gs[0,0])
    ax1.set_title("Input image")
    ax1.imshow(_img_in_RGB)
    ax1.set_xticks([]), ax1.set_yticks([])

    ax2 = fig.add_subplot(gs[0,1])
    ax2.set_title("Histogram equalized image")
    ax2.imshow(_img_in_RGB_equ)
    ax2.set_xticks([]), ax2.set_yticks([])

    ax3 = fig.add_subplot(gs[1,0])
    ax3 = create_RGB_histogram(_img_in_RGB, ax3, "Input image")

    ax4 = fig.add_subplot(gs[1,1])
    ax4 = create_RGB_histogram(_img_in_RGB_equ, ax4, "Histogram equalized image")

    ax3.set_ylim([0, 50000])
    ax4.set_ylim([0, 50000])
    ax3.set_yticks([0, 10000, 20000, 30000, 40000, 50000])
    ax4.set_yticks([0, 10000, 20000, 30000, 40000, 50000])

    plt.show()

def main():
    parser   = argparse.ArgumentParser()
    parser.add_argument('-f', '--filepath', required=True)
    args     = parser.parse_args()
    
    # Read an input image
    img_in_RGB = read_image(args.filepath)
    img_in_YUV = cv2.cvtColor(img_in_RGB, cv2.COLOR_RGB2YUV)
    img_in_HSV = cv2.cvtColor(img_in_RGB, cv2.COLOR_RGB2HSV)

    # Equalize histogram of the input image (YCbCr)
    img_in_YUV[:,:,0] = cv2.equalizeHist(img_in_YUV[:,:,0])
    img_in_RGB_equ_from_YUV = cv2.cvtColor(img_in_YUV, cv2.COLOR_YUV2RGB)

    # Equalize histogram of the input image (HSV)
    img_in_HSV[:,:,2] = cv2.equalizeHist(img_in_HSV[:,:,2])
    img_in_RGB_equ_from_HSV = cv2.cvtColor(img_in_HSV, cv2.COLOR_HSV2RGB)

    # Save the histogram equalized two images
    cv2.imwrite("../IMAGE_DATA/hist_equalized_YUV.bmp", cv2.cvtColor(img_in_RGB_equ_from_YUV, cv2.COLOR_RGB2BGR))
    cv2.imwrite("../IMAGE_DATA/hist_equalized_HSV.bmp", cv2.cvtColor(img_in_RGB_equ_from_HSV, cv2.COLOR_RGB2BGR))

    # Create figure
    create_figure(img_in_RGB, img_in_RGB_equ_from_HSV)

if __name__=="__main__": 
    main()