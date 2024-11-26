import numpy as np
from numba import jit
from scipy import signal
import cv2
import matplotlib.pyplot as plt
import time

# Remove a seam from img
@jit
def remove(img: np.ndarray, seam) -> np.ndarray:
    h, w, c = img.shape
    ret_img = np.zeros((h, w-1, 3))
    for i in range(h):
        m = seam[i][1]
        ret_img[i, :m] = img[i, :m]
        ret_img[i, m:] = img[i, m+1:]
    return ret_img 

@jit 
def find_seam(energy_map: np.ndarray) -> np.ndarray:
    h, w = energy_map.shape
    dp = np.zeros((h, w))

    dp[0, :] = energy_map[0, :]
    for i in range(1, h):
        for j in range(0, w):
            if(j == 0):
                dp[i, j] = min(dp[i-1, j], dp[i-1, j+1]) + energy_map[i, j]
            elif(j==w-1):
                dp[i, j] = min(dp[i-1, j-1], dp[i-1, j]) + energy_map[i, j]
            else:
                dp[i, j] = min(dp[i-1, j-1], min(dp[i-1, j], dp[i-1, j+1])) + energy_map[i, j]

    y_s = np.argmin(dp[h-1, :])
    seam = np.zeros((h, 2), dtype=np.int32)
    seam[-1] = [h-1, y_s]
    for i in range(h-2, -1, -1):
        x, y = seam[-1]
        # print(f"(x,y) = {x, y}")
        if(dp[x, y] == dp[x-1, y-1] + energy_map[x, y]):
            seam[i] = [x-1, y-1]
        elif(dp[x, y] == dp[x-1, y] + energy_map[x, y]):
            seam[i] = [x-1, y-1]
        elif(dp[x, y] == dp[x-1, y+1] + energy_map[x, y]):
            seam[i] = [x-1, y-1]
        else:
            print("Something wrong when compute dp")
            assert(1 == 0)
    return seam


def delete_vertical(img: np.ndarray):

    h, w, c = img.shape

    f_dx = np.array([[0, 0, 0],
                    [1, -2, 1], 
                    [0, 0, 0]])

    f_dy = np.array([[0, 1, 0],
                    [0, -2, 0],
                    [0, 1, 0]])

    convolved_dx = np.zeros((h, w))
    convolved_dy = np.zeros((h, w))
    for channel in range(3):
        dx = signal.convolve2d(img[:, :, channel], f_dx, mode='same', boundary='symm')
        dy = signal.convolve2d(img[:, :, channel], f_dy, mode='same', boundary='symm')
        convolved_dx += dx
        convolved_dy += dy


    energy_map = np.abs(convolved_dx) + np.abs(convolved_dy)

    # dp = np.zeros((h, w))

    # dp[0, :] = energy_map[0, :]
    # for i in range(1, h):
    #     for j in range(0, w):
    #         if(j == 0):
    #             dp[i, j] = min(dp[i-1, j], dp[i-1, j+1]) + energy_map[i, j]
    #         elif(j==w-1):
    #             dp[i, j] = min(dp[i-1, j-1], dp[i-1, j]) + energy_map[i, j]
    #         else:
    #             dp[i, j] = min(dp[i-1, j-1], min(dp[i-1, j], dp[i-1, j+1])) + energy_map[i, j]

    # y_s = np.argmin(dp[h-1, :])
    # seam = np.zeros((h, 2), dtype=np.int32)
    # seam[-1] = [h-1, y_s]
    # for i in range(h-2, -1, -1):
    #     x, y = seam[-1]
    #     # print(f"(x,y) = {x, y}")
    #     if(dp[x, y] == dp[x-1, y-1] + energy_map[x, y]):
    #         seam[i] = [x-1, y-1]
    #     elif(dp[x, y] == dp[x-1, y] + energy_map[x, y]):
    #         seam[i] = [x-1, y-1]
    #     elif(dp[x, y] == dp[x-1, y+1] + energy_map[x, y]):
    #         seam[i] = [x-1, y-1]
    #     else:
    #         print("Something wrong when compute dp")
    #         assert(1 == 0)
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

