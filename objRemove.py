from resize import find_seam, compute_energyMap, remove, show_seam
import numpy as np
from numba import jit
import cv2 as cv
import argparse
import matplotlib.pyplot as plt

def main():
    arg = argparse.ArgumentParser()
    arg.add_argument("-mask", help="Path to mask",required=True)
    arg.add_argument("-image", help="Path to image", required=True)
    args = vars(arg.parse_args())
    mask_path, img_path = args["mask"], args["image"]
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    origin = img
    mask = np.array(cv.imread(mask_path))
    mask = np.where(mask==0, False, True)
    f = np.array([1, -2, 1])
    
    while(np.isin(mask, 0).any()):
        print(np.max(img))
        energyMap = compute_energyMap(img, f)
        
        energyMap = np.where(mask[:, :, 0] == False, 1e-3, energyMap)
        seam = find_seam(energyMap)
        # show_seam(img, seam)
        mask = remove(img=mask, seam=seam)
        img = remove(img=img, seam=seam)
        # print(f"img shape = {img.shape}")
    
    fig, axes = plt.subplots(2, 1, figsize = (9, 10))
    axes[0].imshow(origin)
    axes[0].axis(False)
    axes[0].set_title('Original imaage')
    axes[1].imshow(img)
    axes[1].axis(False)
    axes[1].set_title('After removed object')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()