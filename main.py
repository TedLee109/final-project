import numpy as np
import time
import argparse
import cv2
from resize import resize_image, reszie_by_transport
from enlarge import enlarge_image
from objRemove import obj_remove


def main():
    arg = argparse.ArgumentParser()
    arg.add_argument("-mask", help="Path to mask")
    arg.add_argument("-image", help="Path to image", required=True)
    arg.add_argument("-forward", action="store_true", help="Use forward energy")
    arg.add_argument("-resize", action="store_true", help="whether user want to resize img")
    arg.add_argument("-enlarge", action="store_true")
    arg.add_argument("-numOfDelete_H", help="# horizontal seam to be deleted (insert if neg)")
    arg.add_argument("-numOfDelete_W", help="# vertical seam to be deleted (insert if neg)")
    args = arg.parse_args()

    imgPath, maskPath = args.image, args.mask
    if args.resize or args.enlarge:
        if args.numOfDelete_H is None or args.numOfDelete_W is None:
            arg.error(
                "When using the -resize option, you must also provide both -numOfDelete_H and -numOfDelete_W."
            )
        # Convert to integers
        numOfDelete_H = int(args.numOfDelete_H)
        numOfDelete_W = int(args.numOfDelete_W)
    else:
        numOfDelete_H, numOfDelete_W = None, None

    img = cv2.imread(imgPath)
    if maskPath is not None:
        mask = np.array(cv2.imread(maskPath))
        mask = np.where(mask == 0, False, True)
    else:
        mask = None

    origin_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(origin_img.shape)
    img = origin_img

    if args.resize:
        start_time = time.time()
        img = resize_image(origin_img, numOfDelete_H, numOfDelete_W, args.forward)
        # img = reszie_by_transport(origin_img, numOfDelete_H, numOfDelete_W, args.forward)
        print('Time used: {} sec'.format(time.time() - start_time))
    if args.enlarge:
        start_time = time.time()
        img = enlarge_image(origin_img, numOfDelete_W)
        print('Time used: {} sec'.format(time.time() - start_time))
    if mask is not None:
        start_time = time.time()
        img = obj_remove(origin_img, mask, args.forward)
        print('Time used: {} sec'.format(time.time() - start_time))

    name = imgPath.split("/")[-1]  # Fixed to handle different OS path formats
    output_path = "out/" + name
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, img)


if __name__ == "__main__":
    main()
