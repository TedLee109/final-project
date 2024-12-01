from resize import find_seam, compute_energyMap, remove, show_seam, get_forward_seam
import numpy as np
from numba import jit
import cv2 as cv
import argparse
import matplotlib.pyplot as plt
import time

def update_mask(mask: np.ndarray[np.bool_], seam):
    h, w = mask.shape[:2]
    ret = np.zeros((h, w-1, 3), np.bool_)
    for i in range(h):
        m = seam[i]
        # print(m)
        ret[i, :m] = mask[i, :m]
        ret[i, m:] = mask[i, m+1:]
    return ret

def main():
    arg = argparse.ArgumentParser()
    arg.add_argument("-mask", help="Path to mask",required=True)
    arg.add_argument("-image", help="Path to image", required=True)
    arg.add_argument("-forward", action="store_true", help="Use forward energy")
    args = vars(arg.parse_args())
    mask_path, img_path, forward = args["mask"], args["image"], args["forward"]
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    origin = img
    mask = np.array(cv.imread(mask_path))
    mask = np.where(mask==0, False, True)
    f = np.array([1, -2, 1])
    start_t = time.time()
    while(np.isin(mask, False).any()):
        if forward:
            seam = get_forward_seam(img, mask)
        else:
            energyMap = compute_energyMap(img, f)
            energyMap[mask[:, :, 0] == False] = -1e6
            seam = find_seam(energyMap)
        # show_seam(img, seam)
        mask = update_mask(mask=mask, seam=seam)
        img = remove(img=img, seam=seam)
    print('Time used: {} sec'.format(time.time()-start_t))
    fig, axes = plt.subplots(2, 1, figsize = (9, 10))
    axes[0].imshow(origin)
    axes[0].axis(False)
    axes[0].set_title('Original imaage')
    axes[1].imshow(img)
    axes[1].axis(False)
    axes[1].set_title('After removed object')
    plt.tight_layout()
    plt.show()
    name = img_path.split("/")[1]
    output_path = "out/" + name; 
    img = img.astype(np.uint8)
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    cv.imwrite(output_path, img)


if __name__ == "__main__":
    main()