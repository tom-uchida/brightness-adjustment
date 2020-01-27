# ガンマ補正プログラム
# https://www.dogrow.net/python/blog99/

# Example: 
# python3 gamma_correct_test.py -f input_img.png -g 0.8

import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
plt.style.use('bmh')

def exec_gamma_correction( _filepath, _gamma ):
    # Read an input image
    img_in = cv2.imread(_filepath)

    # Create LUT by using gamma
    LUT = np.empty( (1,256), np.uint8 )
    for i in range(256):
        LUT[0,i] = np.clip(pow(i/255.0, _gamma) * 255.0, 0, 255)

    # Gamma correct pixel value with LUT
    img_corrected = cv2.LUT(img_in, LUT)

    # Save the corrected image
    cv2.imwrite("../IMAGE_DATA/gamma_"+str(_gamma)+".bmp", img_corrected)

    create_figure(LUT)

def create_figure(_LUT):
    fig = plt.figure(figsize=(8, 6)) # figsize=(width, height)
    gs  = gridspec.GridSpec(1,1)

    ax = fig.add_subplot(gs[0,0])
    # ax.set_title('Tone Curve')
    # ax.set_xlabel('Input pixel value', fontsize=28)
    # ax.set_ylabel('Output pixel value', fontsize=28)
    ax.set_xlabel('', fontsize=18)
    ax.set_ylabel('', fontsize=18)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=14)

    ax.plot(np.arange(256), _LUT[0,:], color='black')

    plt.show()

def main():
    parser   = argparse.ArgumentParser()
    parser.add_argument('-g',   '--gamma',      required=True)
    parser.add_argument('-f',   '--filepath',   required=True)
    args     = parser.parse_args()
    
    # Do gamma correction
    exec_gamma_correction( args.filepath, float(args.gamma) )

if __name__=="__main__": 
    main()