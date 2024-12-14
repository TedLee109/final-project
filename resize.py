import numpy as np
from numba import jit
from scipy import signal
from scipy import ndimage as ndi
import cv2
import matplotlib.pyplot as plt
import time
import argparse
from tqdm import tqdm

def show_seam(img: np.ndarray, seams):

    for seam in seams:
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
    ret_img = np.zeros((h, w-1, 3), dtype=img.dtype)
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
    S = np.arange(-1, w-1)
    dp[0, :] = energy_map[0, :]
    for i in range(1, h):
        # row = dp[i-1]
        # L = np.roll(row, 1)
        # L[0] = 1e8
        # R = np.roll(row, -1)
        # R[w-1] = 1e8
        # LUR = np.array([L, row, R])
        # idxs = np.argmin(LUR, axis=0)
        # dp[i] = np.choose(idxs, LUR) + energy_map[i]
        # prev[i] = idxs + S
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

@jit 
def find_seam_energy(energy_map: np.ndarray) -> np.ndarray:
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
    return seam, np.min(dp[h-1])

def calculate_energy(image):
    """計算圖像的能量圖，對每個色彩通道計算梯度幅值並合併"""
    
    channels = cv2.split(image)
    
    energy = np.zeros(image.shape[:2], dtype=np.float64)
    
    for channel in channels:
        sobel_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
        energy += np.abs(sobel_x) + np.abs(sobel_y)
    
    return energy

def compute_energyMap(f: np.ndarray, g: np.ndarray = np.array([-1, 0, 1])):

    dx = ndi.convolve1d(f, g, axis=0, mode='mirror')
    dy = ndi.convolve1d(f, g, axis=1, mode='mirror')
    ret = np.sqrt(np.sum(dx ** 2, axis=2) + np.sum(dy ** 2, axis=2))
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
    
    temp_image = img.copy()
    if forward:
        seam = get_forward_seam(img)
    else :
        energy_map = compute_energyMap(temp_image)
        seam = find_seam(energy_map=energy_map)
    ret = remove(img=img, seam=seam)
    return ret

def delete_horizontal(img: np.ndarray, forward):
    img = np.transpose(img, (1, 0, 2))
    ret = delete_vertical(img, forward)
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

def reszie_by_transport(img: np.ndarray, delete_height: int, delete_width: int, forward: bool):
    h, w, _ = img.shape
    transportMap = np.zeros((h, w,))
    tmp_images = [None] * delete_height
    for i in tqdm(range(delete_width)):
        for j in range(delete_height):
            if i == 0 and j == 0:
                tmp_images[j] = img
                continue; 
            elif i == 0:
                horizontal = compute_energyMap(tmp_images[j-1])
                hor_seam, hor_e = find_seam_energy(horizontal)
                tmp_images[j] = remove(tmp_images[j-1], hor_seam)
                transportMap[i, j] = transportMap[i, j-1] + hor_e
                continue
            elif j == 0:
                vertical = compute_energyMap(tmp_images[j])
                ver_seam, ver_e = find_seam_energy(vertical)
                tmp_images[j] = remove(tmp_images[j], ver_seam)
                transportMap[i, j] = transportMap[i-1, j] + ver_e
                continue

            vertical = compute_energyMap(tmp_images[j])
            horizontal = compute_energyMap(tmp_images[j-1])
            ver_seam, ver_e = find_seam_energy(vertical)
            hor_seam, hor_e = find_seam_energy(horizontal)
            transportMap[i, j] = min(transportMap[i-1, j]+ver_e, transportMap[i, j-1]+hor_e)
            if transportMap[i-1, j]+ver_e < transportMap[i, j-1]+hor_e:
                tmp_images[j] = remove(tmp_images[j], ver_seam)
            else:
                tmp_images[j] = remove(tmp_images[j-1], hor_seam)
    return tmp_images[delete_height - 1]


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