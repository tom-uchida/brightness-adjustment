import cv2
import numpy as np
import sys
args = sys.argv

# Check arguments
import sys
args = sys.argv
if len(args) != 3:
    print( "\nUSAGE   : $ python {} [input_image_name] [output_image_name]".format ( args[0] ) )
    print( "EXAMPLE : $ python {}\n".format( args[0] ) )
    sys.exit()

def main():
    # Read the input image
    img = cv2.imread( args[1] )

    # Get height and width
    height, width = img.shape[0], img.shape[1]

    # User input
    print('Please input start point for cropping:')
    start_x = input('start_x(0 <= x <= ' + str(width)  + '): ')
    start_y = input('start_y(0 <= y <= ' + str(height) + '): ')
    print('\nPlease input height and width for cropping:')
    w = input('width : ')
    h = input('height: ')

    # 画像のクロッピング
    crop_image = img[int(start_y):int(start_y)+int(h), int(start_x):int(start_x)+int(w)]

    # 出力ファイル名
    crop_image_name = args[2]
    cv2.imwrite( crop_image_name, crop_image )

if __name__ == "__main__":
    main()