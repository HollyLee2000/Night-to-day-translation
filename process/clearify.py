import numpy as np
import cv2

# This program can make images clearer by simple methods.
img = cv2.imread('castle_dark_defog.jpg')
index = [2, 1, 0]
img = img[:, :, index]  # turn BGR into RGB
rgbscale = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# edge_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
img = cv2.filter2D(rgbscale, -1, sharpen_kernel)
cv2.imwrite('castle_dark_defog_sharp.png', img)
cv2.imwrite('castle_dark_defog_sharp.jpg', img)

# Smooth out image
blur2 = cv2.medianBlur(img, 3)
blur = cv2.GaussianBlur(img, (3, 3), 0)
blur3 = cv2.bilateralFilter(img, 3, 400, 400)
cv2.imwrite('castle_dark_defog_sharp_gauss.png', blur)
cv2.imwrite('castle_dark_defog_sharp_median.png', blur2)
cv2.imwrite('castle_dark_defog_sharp_bilatera.png', blur3)
cv2.imwrite('castle_dark_defog_sharp_gauss.jpg', blur)
cv2.imwrite('castle_dark_defog_sharp_median.jpg', blur2)
cv2.imwrite('castle_dark_defog_sharp_bilatera.jpg', blur3)

