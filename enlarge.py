import cv2
import time
import numpy as np
from resize import find_seam

def calculate_energy(image):
    """計算圖像的能量圖，使用 Sobel 算子計算梯度幅值"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    energy = np.abs(sobel_x) + np.abs(sobel_y)
    return energy

def add_seams(image, seams):
    """Add multiple seams to the image based on the provided list of seams."""
    rows, cols, channels = image.shape
    enlarged = np.zeros((rows, cols + len(seams), channels), dtype=image.dtype)

    new_seams = np.array([seams.pop()])
    for s in reversed(seams): 
        new_seams[np.where(new_seams >= s)] += 1
        new_seams = np.append(new_seams, s.reshape((1, -1)), axis=0) 
    
    for i in range(rows):
        # Sort seams for this row to maintain the correct order of duplication
        sorted_seams = sorted(new_seams, key=lambda x: x[i])
        col_offset = 0
        for j in range(cols):
            enlarged[i, j + col_offset] = image[i, j]
            while col_offset < len(sorted_seams) and sorted_seams[col_offset][i] == j:
                # Duplicate the seam pixel by averaging with its neighbors
                for ch in range(channels):
                    if j == 0:
                        enlarged[i, j + col_offset + 1, ch] = (image[i, j, ch].astype(np.int32) + image[i, j + 1, ch].astype(np.int32)) // 2
                    elif j == cols - 1:
                        enlarged[i, j + col_offset + 1, ch] = (image[i, j - 1, ch].astype(np.int32) + image[i, j, ch].astype(np.int32)) // 2
                    else:
                        enlarged[i, j + col_offset + 1, ch] = (image[i, j - 1, ch].astype(np.int32) + image[i, j, ch].astype(np.int32)) // 2
                col_offset += 1
    
    return enlarged

def find_multiple_seams(image, num_seams):
    """Find multiple seams for removal simultaneously."""
    seams = []
    temp_image = image.copy()
    for _ in range(num_seams):
        energy_map = calculate_energy(temp_image)
        seam = find_seam(energy_map)
        seams.append(seam)
        # Remove the seam from the temporary image to avoid selecting the same seam repeatedly
        temp_image = remove_seam(temp_image, seam)
    return seams

def remove_seam(image, seam):
    """Remove a single seam from the image."""
    rows, cols, channels = image.shape
    reduced = np.zeros((rows, cols - 1, channels), dtype=image.dtype)
    
    for i in range(rows):
        col = seam[i]
        reduced[i, :col] = image[i, :col]
        reduced[i, col:] = image[i, col + 1:]
    
    return reduced

def enlarge_image(image, num_seams):
    """Enlarge the image by finding and duplicating multiple seams."""

    # Find multiple seams for removal
    start = time.time()
    seams = find_multiple_seams(image, num_seams)
    print('Time used: {} sec to find multiple seams'.format(time.time() - start))
    start = time.time()
    # Add the seams back to enlarge the image
    enlarged_image = add_seams(image, seams)
    print('Time used: {} sec to add seams'.format(time.time() - start))

    return enlarged_image