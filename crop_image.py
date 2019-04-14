import cv2
import numpy as np
import sys
args = sys.argv

def main():
    # Read image
    image_name = args[1]
    img = cv2.imread( image_name )

    # Get height and width of inout image
    height, width = img.shape[0], img.shape[1]

    # User input
    print('クロッピングの開始位置を入力してください')
    start_x = input('x座標(0 <= x <= ' + str(width)  + ') >>')
    start_y = input('y座標(0 <= y <= ' + str(height) + ') >>')
    print('\nクロッピングする幅・高さを入力してください')
    w = input('幅　 >>')
    h = input('高さ >>')

    # 画像のクロッピング
    crop_image = img[int(start_y):int(start_y)+int(h), int(start_x):int(start_x)+int(w)]

    # 出力ファイル名
    crop_image_name = args[2]
    cv2.imwrite( crop_image_name, crop_image )

if __name__ == "__main__":
    main()