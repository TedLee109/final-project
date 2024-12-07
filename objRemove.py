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

def obj_remove(img:np.ndarray, mask: np.ndarray, forward: bool):
    f = np.array([1, -2, 1])
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
    return img