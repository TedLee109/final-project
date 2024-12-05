import numpy as np
from numba import jit
from scipy import signal
from scipy import ndimage as ndi
import cv2
import matplotlib.pyplot as plt
import time
import argparse

def show_seam(img: np.ndarray, seam):
    h, w = img.shape[:2]
    
    for x, Si in enumerate(seam):
        y = Si
        img[x, y, :] = [255, 0, 0]
    
    plt.imshow(img)
    plt.axis(False)
    plt.show()

# Remove a seam from img
# @jit
def remove(img: np.ndarray, seam) -> np.ndarray:
    h, w = img.shape[:2]
    ret_img = np.zeros((h, w-1, 3), np.int32)
    for i in range(h):
        m = seam[i]
        # print(m)
        ret_img[i, :m] = img[i, :m]
        ret_img[i, m:] = img[i, m+1:]
    return ret_img 

@jit 
def find_seam(energy_map: np.ndarray) -> np.ndarray:
    h, w = energy_map.shape
    dp = np.zeros((h, w), dtype=np.int64)
    prev = np.zeros((h, w))
    dp[0, :] = energy_map[0, :]
    for i in range(1, h):
        for j in range(0, w):
            l = max(0, j-1)
            r = min(w-1, j+1)
            y = np.argmin(dp[i-1, l:r+1]) + l
            dp[i, j] = dp[i-1, y] + energy_map[i, j]
            prev[i, j] = y

    y_s = np.argmin(dp[h-1, :])
    seam = np.zeros(h, dtype=np.int32)
    seam[-1] = y_s
    for i in range(h-2, -1, -1):
        seam[i] = prev[i+1, seam[i+1]]
    return seam


def compute_energyMap(f: np.ndarray, g: np.ndarray):

    dx = ndi.convolve1d(f, g, axis=0, mode='mirror').sum(axis=2)
    dy = ndi.convolve1d(f, g, axis=1, mode='mirror').sum(axis=2)
    ret = np.array(np.abs(dx) + np.abs(dy), dtype=np.int64)
    return ret


@jit 
def get_forward_seam(I: np.ndarray, mask: np.ndarray = None):
    I = I.sum(axis=2, dtype=np.int64)
    h, w = I.shape
    M = np.zeros((h, w), np.int64)
    prev = np.zeros((h, w), dtype=np.int32)
    for i in range(1, h):
        for j in range(w):
            l = max(0, j-1)
            r = min(w-1, j+1)
            cU = np.abs(I[i, r] - I[i, l])
            costs = np.array([np.abs(I[i-1, j] - I[i, l])+cU, cU, np.abs(I[i-1,j] - I[i, r])+cU], np.int64)
            sum_M = np.array([M[i-1, l], M[i-1, j], M[i-1, r]], np.int64) + costs
            idx = np.argmin(sum_M)
            if mask is not None:
                p = -1e6 if not mask[i, j, 0] else 0
            else:
                p = 0
            
            M[i, j] = sum_M[idx] + p
            pos = np.array([l, j, r])
            prev[i, j] = pos[idx]
    seams = np.zeros(h, np.int32)
    seams[-1] = np.argmin(M[h-1, :])
    for i in range(h-2, -1, -1):
        seams[i] = prev[i+1, seams[i+1]]
    return seams


def delete_vertical(img: np.ndarray, forward: bool):

    f = np.array([1, -2, 1])
    if forward:
        seam = get_forward_seam(img)
    else :
        energy_map = compute_energyMap(img, f)
        seam = find_seam(energy_map=energy_map)
    ret = remove(img=img, seam=seam)
    return ret

def delete_horizontal(img: np.ndarray):
    img = np.transpose(img, (1, 0, 2))
    ret = delete_vertical(img)
    return np.transpose(ret, (1, 0, 2))

def resize_image(img: np.ndarray, delete_height: int, delete_width: int, forward: bool):
    h, w, _ = img.shape
    resized_img = img
    while delete_height > 0 or delete_width > 0:
        if delete_height > 0 and (delete_width == 0 or delete_height >= delete_width):
            resized_img = delete_horizontal(resized_img, forward)
            delete_height -= 1
        elif delete_width > 0:
            resized_img = delete_vertical(resized_img, forward)
            delete_width -= 1
    return resized_img

def show_result(origin_img, img):
    img = np.int32(img)
    fig, axes = plt.subplots(2, 1, figsize = (9, 10))
    axes[0].imshow(origin_img)
    axes[0].axis(False)
    axes[0].set_title('origin image')
    axes[1].imshow(img)
    axes[1].axis(False)
    axes[1].set_title(f"Result")
    plt.tight_layout()
    plt.show()