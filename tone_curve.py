import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
#import seaborn as sns
plt.style.use('seaborn-white')

from matplotlib import cycler
colors = cycler('color', ['#EE6666', '#3388BB', '#9988DD', '#EECC55', '#88BB44', '#FFBBBB'])
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True, grid=False, prop_cycle=colors)
plt.rc('grid', color='w', linestyle='solid')
plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=2)



# ------------------------
# ----- Plot Setting -----
# ------------------------
def rgb_hist(rgb_img, ax, ticks=None):
    """
    rgb_img と matplotlib.axes を受け取り、
    axes にRGBヒストグラムをplotして返す
    """
    color=['r','g','b']
    for (i,col) in enumerate(color): 
        hist = cv2.calcHist([rgb_img], [i], None, [256], [0,256])
        hist = np.sqrt(hist)
        ax.plot(hist,color=col)
        
    if ticks:
        ax.set_xticks(ticks)
        
    ax.set_title('Histogram')
    ax.set_xlim([-5,260])
    
    return ax

def plot_curve(f, rgb_img):
    fig = plt.figure(figsize=(12,7))
    gs = gridspec.GridSpec(2,3)
    x = np.arange(256)
    
    # tone curve (0,1)&(1,1)
    #sns.set_style('darkgrid')
    ax2 = fig.add_subplot(gs[:,1]) # 2列目
    ax2.set_title('Tone Curve')    
    #ticks = [0,63,127,191,255]
    ax2.set_xlabel('Input pixel value')
    ax2.set_ylabel('Output pixel value')
    # ax2.set_xticks(ticks)
    # ax2.set_yticks(ticks)
    ax2.set_aspect('equal')
    ax2.plot(x, f(x))
     
    # Input image (0,0)
    #sns.set_style('ticks')
    ax1 = fig.add_subplot(gs[0,0])
    ax1.set_title('Input image')
    ax1.imshow(rgb_img)
    ax1.set_xticks([]), ax1.set_yticks([]) # off scale
    
    # Convert pixel value
    # note:uint8
    #out_rgb_img = np.array([f(a).astype('uint8') for a in rgb_img]) 
    LUT = np.arange(256, dtype='uint8').reshape(-1,1)
    LUT = np.array([f(a).astype('uint8') for a in LUT])
    out_rgb_img = cv2.LUT(rgb_img, LUT)

    # Output image
    ax3 = fig.add_subplot(gs[0,2])
    ax3.set_title('Output image')
    ax3.imshow(out_rgb_img)
    ax3.set_xticks([]), ax3.set_yticks([])
    
    # Histogram
    #sns.set_style(style='whitegrid')
    ax4 = fig.add_subplot(gs[1,0])
    #ax4 = rgb_hist(rgb_img, ax4, ticks)
    ax4 = rgb_hist(rgb_img, ax4)
    ax5 = fig.add_subplot(gs[1,2])
    #ax5 = rgb_hist(out_rgb_img, ax5, ticks)
    ax5 = rgb_hist(out_rgb_img, ax5)
    
    plt.show()


# ----------------------------
# ----- Read Input Image -----
# ----------------------------
def read_img(_img_name):
  # read input image
  img_BGR = cv2.imread(_img_name)

  # convert color (BGR → RGB)
  img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

  return img_RGB

img_in_RGB = read_img('images/2018-10-01/funehoko200.jpg')
print("Input image(RGB) : ", img_in_RGB.shape) # （height × width × 色数）
print("\n")



# -------------------------------
# ----- Tone Curve Function -----
# -------------------------------
def curve_1(_x):
    y = 3*_x
    return y

def curve_2(_x):
    y = (np.sin(np.pi * (_x/255 - 0.5)) + 1)/2 * 255
    return y

def curve_gamma(_x):
    gamma = 3 # 2, 1/2
    y = 255*(_x/255)**(1/gamma)
    return y




# ---------------------
# ----- Execution -----
# ---------------------
#plot_curve(curve_1, img_in_RGB)
#plot_curve(curve_2, img_in_RGB)
plot_curve(curve_gamma, img_in_RGB)
#plot_curve(curve_gamma, img_in_RGB)