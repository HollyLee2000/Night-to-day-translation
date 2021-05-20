import os
import cv2
import numpy as np
import json
import retinex
import defog

# Batch processing of images clearify
data_path = 'clearify_in/'
img_list = os.listdir(data_path)
out_path = 'clearify_out/'
if len(img_list) == 0:
    print('Data directory is empty.')
    exit()

for img_name in img_list:
    if img_name == '.gitkeep':
        continue

    img2 = cv2.imread(os.path.join(data_path, img_name))
    name = img_name[:-4]
    print('clearifying processing......')


    index = [2, 1, 0]
    img2 = img2[:, :, index]
    rgbscale = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # edge_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img = cv2.filter2D(rgbscale, -1, sharpen_kernel)
    # cv2.imwrite('castle_dark_defog_sharp.png', img)
    # cv2.imwrite('MSRCP_dark_defog_sharp.jpg', img)

    # Smooth out image
    # blur2 = cv2.medianBlur(img, 3)
    # blur = cv2.GaussianBlur(img, (3, 3), 0)
    blur3 = cv2.bilateralFilter(img, 3, 400, 400)
    cv2.imwrite(out_path + name + "_clearify.jpg", blur3)
    # cv2.imwrite('castle_dark_defog_sharp_gauss.png', blur)
    # cv2.imwrite('castle_dark_defog_sharp_median.png', blur2)
    # cv2.imwrite('castle_dark_defog_sharp_bilatera.png', blur3)
    # cv2.imwrite('castle_dark_defog_sharp_gauss.jpg', blur)
    # cv2.imwrite('castle_dark_defog_sharp_median.jpg', blur2)
    # cv2.imwrite(name+'_MSRCP_dark_defog_sharp_bilatera.jpg', blur3)
    print("Done!")

'''
    m = dark_channel_defog.deHaze(img_msrcp / 255.0) * 255
    # cv2.imwrite(name+'_MSRCP_dark_defog.jpg', m)
    img2 = cv2.imread(name+'_MSRCP_dark_defog.jpg')
    index = [2, 1, 0]
    img2 = img2[:, :, index]  # BGR转RGB
    rgbscale = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # edge_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 
    img = cv2.filter2D(rgbscale, -1, sharpen_kernel)
    # cv2.imwrite('castle_dark_defog_sharp.png', img)
    # cv2.imwrite('MSRCP_dark_defog_sharp.jpg', img)

    # Smooth out image
    # blur2 = cv2.medianBlur(img, 3)
    # blur = cv2.GaussianBlur(img, (3, 3), 0)
    blur3 = cv2.bilateralFilter(img, 3, 400, 400)
    # cv2.imwrite('castle_dark_defog_sharp_gauss.png', blur)
    # cv2.imwrite('castle_dark_defog_sharp_median.png', blur2) 
    # cv2.imwrite('castle_dark_defog_sharp_bilatera.png', blur3)
    # cv2.imwrite('castle_dark_defog_sharp_gauss.jpg', blur)
    # cv2.imwrite('castle_dark_defog_sharp_median.jpg', blur2) 
    # cv2.imwrite(name+'_MSRCP_dark_defog_sharp_bilatera.jpg', blur3)
'''
