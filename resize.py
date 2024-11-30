import numpy as np
from numba import jit
from scipy import signal
from scipy import ndimage as ndi
import cv2
import matplotlib.pyplot as plt
import time

def show_seam(img: np.ndarray, seam):
    h, w = img.shape[:2]
    
    for Si in seam:
        (x, y) = Si
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
        m = seam[i][1]
        # print(m)
        ret_img[i, :m] = img[i, :m]
        ret_img[i, m:] = img[i, m+1:]
    return ret_img 

@jit 
def find_seam(energy_map: np.ndarray) -> np.ndarray:
    h, w = energy_map.shape
    dp = np.zeros((h, w), dtype=np.int64)
    prev = np.zeros((h, w, 2))
    dp[0, :] = energy_map[0, :]
    for i in range(1, h):
        for j in range(0, w):
            y = -1
            if(j == 0):
                y = np.argmin(dp[i-1, j:j+2]) + j
            elif(j==w-1):
                y = np.argmin(dp[i-1, j-1:j+1]) + j-1
            else:
                y = np.argmin(dp[i-1, j-1:j+2]) + j-1
            dp[i, j] = dp[i-1, y] + energy_map[i, j]
            prev[i, j] = [i-1, y]

    y_s = np.argmin(dp[h-1, :])
    seam = np.zeros((h, 2), dtype=np.int32)
    seam[-1] = [h-1, y_s]
    for i in range(h-2, -1, -1):
        x = seam[i+1][0]
        y = seam[i+1][1]
        seam[i] = prev[x, y]
    return seam


def compute_energyMap(f: np.ndarray, g: np.ndarray):

    dx = ndi.convolve1d(f, g, axis=0, mode='mirror').sum(axis=2)
    dy = ndi.convolve1d(f, g, axis=1, mode='mirror').sum(axis=2)
    ret = np.array(np.abs(dx) + np.abs(dy), dtype=np.int64)
    return ret

    

def delete_vertical(img: np.ndarray):

    f = np.array([1, -2, 1])

    energy_map = compute_energyMap(img, f)
    assert(energy_map.shape == img.shape[:2])

    seam = find_seam(energy_map=energy_map)
    ret = remove(img=img, seam=seam)
    return ret

def main():
    start_time = time.time()
    img = cv2.imread('image/cat.jpg')
    numOfDelete = 10
    origin_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = origin_img
    for i in range(numOfDelete):
        # print(f"img[50, 10:20] = {img[50, 10:20]}")
        img = delete_vertical(img)
    print('Time used: {} sec'.format(time.time()-start_time))
    img = np.int32(img)
    fig, axes = plt.subplots(2, 1, figsize = (9, 10))
    axes[0].imshow(origin_img)
    axes[0].axis(False)
    axes[0].set_title('origin image')
    axes[1].imshow(img)
    axes[1].axis(False)
    axes[1].set_title(f"Result of {numOfDelete} deletion")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

