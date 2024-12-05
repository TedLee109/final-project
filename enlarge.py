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
    
    for i in range(rows):
        # Sort seams for this row to maintain the correct order of duplication
        sorted_seams = sorted(seams, key=lambda x: x[i])
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

def enlarge_image(image, scale_factor):
    """Enlarge the image by finding and duplicating multiple seams."""
    rows, cols, _ = image.shape
    target_cols = int(cols * scale_factor)
    num_seams = target_cols - cols

    # Find multiple seams for removal
    seams = find_multiple_seams(image, num_seams)
    
    # Add the seams back to enlarge the image
    enlarged_image = add_seams(image, seams)

    return enlarged_image

start_time = time.time()
# 測試
image = cv2.imread('image/dolphin.jpg')  # 載入圖像
scale_factor = 1.5  # 放大比例
enlarged_image = enlarge_image(image, scale_factor)
print('Elapsed time: %.2f seconds' % (time.time() - start_time))
# 保存結果
cv2.imwrite('enlarged_image.jpg', enlarged_image)