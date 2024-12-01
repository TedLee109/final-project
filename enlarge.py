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

def add_seam(image, seam):
    """插入單條接縫"""
    rows, cols, channels = image.shape
    enlarged = np.zeros((rows, cols + 1, channels), dtype=image.dtype)
    for i in range(rows):
        col = seam[i, 1]
        for ch in range(channels):
            if col == 0:
                new_pixel = image[i, col, ch]
            else:
                # 確保使用高精度類型進行運算，並避免溢出
                new_pixel = (int(image[i, col - 1, ch]) + int(image[i, col, ch])) // 2
                new_pixel = np.clip(new_pixel, 0, 255)  # 防止轉回 uint8 時溢出
            enlarged[i, :col, ch] = image[i, :col, ch]
            enlarged[i, col, ch] = new_pixel
            enlarged[i, col + 1:, ch] = image[i, col:, ch]
    return enlarged


def enlarge_image(image, scale_factor):
    """放大圖像，根據比例計算需要插入的接縫數"""
    rows, cols, _ = image.shape
    target_cols = int(cols * scale_factor)
    num_seams = target_cols - cols

    for _ in range(num_seams):
        energy_map = calculate_energy(image)
        seam = find_seam(energy_map)
        image = add_seam(image, seam)

    return image

start_time = time.time()
# 測試
image = cv2.imread('image/cat.jpg')  # 載入圖像
scale_factor = 1.5  # 放大比例
enlarged_image = enlarge_image(image, scale_factor)
print('Elapsed time: %.2f seconds' % (time.time() - start_time))
# 保存結果
cv2.imwrite('enlarged_image.jpg', enlarged_image)