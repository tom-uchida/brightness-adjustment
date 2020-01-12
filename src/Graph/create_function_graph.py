import cv2, matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.style.use('bmh')

# from matplotlib import cycler
# colors = cycler('color', ['#EE6666', '#3388BB', '#9988DD', '#EECC55', '#88BB44', '#FFBBBB'])
# plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True, grid=False, prop_cycle=colors)
# plt.rc('grid', color='w', linestyle='solid')
# plt.rc('patch', edgecolor='#E6E6E6')
# plt.rc('lines', linewidth=2)
# plt.rcParams["mathtext.fontset"] = "cm"
# plt.rcParams["mathtext.rm"] = "Times New Roman"

def tone_curve(_x, _p):
    y = np.where(_x < 255/_p, _p*_x, 255)
    return y

def create_figure():
    fig = plt.figure(figsize=(8, 8)) # figsize=(width, height)
    gs  = gridspec.GridSpec(1,1)

    ax = fig.add_subplot(gs[0,0])
    # ax.set_title('Tone Curve')
    ax.set_xlabel('Input pixel value', fontsize=18)
    ax.set_ylabel('Output pixel value', fontsize=18)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=18)
    # ax.set_xlim([-1, 256])
    # ax.set_ylim([-1, 256])
    ax.plot(x, tone_curve(x, p), color='black')

    plt.show()

if __name__ == "__main__":
    x = np.arange(256)
    p = 2.52

    create_figure()