import cv2
import time
import numpy as np
from resize import find_seam

def calculate_energy(image):
    """Calculate the energy map of the image using the Sobel operator."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    energy = np.abs(sobel_x) + np.abs(sobel_y)
    return energy

def add_seam(image, seam):
    """Insert a single seam into the image."""
    rows, cols, channels = image.shape
    enlarged = np.zeros((rows, cols + 1, channels), dtype=image.dtype)
    for i in range(rows):
        col = seam[i, 1]
        for ch in range(channels):
            if col == 0:
                new_pixel = image[i, col, ch]
            else:
                # Ensure high precision to avoid overflow
                new_pixel = (int(image[i, col - 1, ch]) + int(image[i, col, ch])) // 2
                new_pixel = np.clip(new_pixel, 0, 255)  # Prevent overflow in uint8
            enlarged[i, :col, ch] = image[i, :col, ch]
            enlarged[i, col, ch] = new_pixel
            enlarged[i, col + 1:, ch] = image[i, col:, ch]
    return enlarged

def enlarge_image(image, scale_factor):
    """Enlarge the image by inserting seams."""
    rows, cols, _ = image.shape
    target_cols = int(cols * scale_factor)
    num_seams = target_cols - cols

    seams = []  # Store the seams inserted to avoid duplication

    for _ in range(num_seams):
        energy_map = calculate_energy(image)

        # Update energy map to avoid duplicating previous seams
        for seam in seams:
            for row in range(rows):
                energy_map[row, seam[row, 1]] = float('inf')  # Avoid previously inserted seams

        seam = find_seam(energy_map)  # Find the new seam
        seams.append(seam)  # Save the current seam
        image = add_seam(image, seam)  # Add the seam to the image

    return image

def enlarge_image_with_steps(image, scale_factor, step_factor=0.5):
    """
    Enlarge the image in steps to improve quality.
    Instead of inserting all seams at once, perform seam insertion in multiple stages.
    """
    current_scale = 1.0
    while current_scale < scale_factor:
        step_scale = min(step_factor, scale_factor - current_scale)
        image = enlarge_image(image, current_scale + step_scale)
        current_scale += step_scale
    return image

# Test the updated function
start_time = time.time()
image = cv2.imread('image/cat.jpg')  # Load image
scale_factor = 1.5  # Target enlargement factor
step_factor = 0.5  # Intermediate step scale

# Enlarge the image with multiple steps for better quality
enlarged_image = enlarge_image_with_steps(image, scale_factor, step_factor)
print('Elapsed time: %.2f seconds' % (time.time() - start_time))

# Save the result
cv2.imwrite('enlarged_image.jpg', enlarged_image)