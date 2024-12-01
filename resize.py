import numpy as np
from numba import jit
from scipy import signal
from scipy import ndimage as ndi
import cv2
import matplotlib.pyplot as plt
import time
from skimage import color

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


def compute_forward_energy(I: np.ndarray):
    I = I.sum(axis=2, dtype=np.int64)
    h, w = I.shape
    M = np.zeros((h, w))

    L = np.roll(I, 1, axis=1)
    R = np.roll(I, -1, axis=1)
    U = np.roll(I, 1, axis=0)

    CU = np.abs(R - L)
    CL = np.abs(U - L) + CU
    CR = np.abs(U - R) + CU
    
    energy = np.zeros((h, w))
    for i in range(1, h):
        mL = np.roll(M[i-1], 1)
        mR = np.roll(M[i-1], -1)
        costs = np.array([CL[i], CU[i], CR[i]])
        sum_M = np.array([mL, M[i-1], mR]) + costs
        mins = np.argmin(sum_M, axis=0)
        energy[i] = np.choose(mins, costs)
        M[i] = np.choose(mins, costs)
    return energy



def delete_vertical(img: np.ndarray):

    f = np.array([1, -2, 1])

    energy_map = compute_energyMap(img, f)
    # energy_map = compute_forward_energy(img)
    assert(energy_map.shape == img.shape[:2])

    seam = find_seam(energy_map=energy_map)
    ret = remove(img=img, seam=seam)
    return ret

def main():
    start_time = time.time()
    img = cv2.imread('image/bench3.png')
    numOfDelete = 200
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
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("out/bench3_backward.jpg", img)

if __name__ == "__main__":
    main()

