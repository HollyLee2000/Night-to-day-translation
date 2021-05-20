import numpy as np
import cv2
import os

# The code for adjust the lightness and saturation of images

# Adjust the maximum
MAX_VALUE = 100


def update(input_img_path, output_img_path, rgb_dir):
    """
    :param input_img_path: input path
    :param output_img_path: output path
    :param lightness: lightness
    :param saturation: saturation
    """

    image = cv2.imread(input_img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
    img2 = cv2.imread(input_img_path)
    # BGR to HLS
    hlsImg = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    avg = np.mean(hlsImg[:, :, 1])
    max = np.max(hlsImg[:, :, 1])
    for i in range(hlsImg[:, :, 1].shape[0]):
        for j in range(hlsImg[:, :, 1].shape[1]):
            # print(avg)
            # print(max)
            if hlsImg[:, :, 1][i][j] > 0.9:
                hlsImg[:, :, 1][i][j] = hlsImg[:, :, 1][i][j] / 2
                hlsImg[:, :, 2][i][j] = hlsImg[:, :, 2][i][j] / 2
                img2[i][j][0] = img2[i][j][0] / 2
                img2[i][j][1] = img2[i][j][1] / 2
                img2[i][j][2] = img2[i][j][2] / 2
    hlsImg[:, :, 1][hlsImg[:, :, 1] > 1] = 1
    # Adjust brightness (linear transformation)
    # hlsImg[:, :, 1] = (1.0 + lightness / float(MAX_VALUE)) * hlsImg[:, :, 1]
    # hlsImg[:, :, 1][hlsImg[:, :, 1] > 1] = 1
    # saturation
    # hlsImg[:, :, 2] = (1.0 + saturation / float(MAX_VALUE)) * hlsImg[:, :, 2]
    hlsImg[:, :, 2][hlsImg[:, :, 2] > 1] = 1
    # HLS2BGR
    lsImg = cv2.cvtColor(hlsImg, cv2.COLOR_HLS2BGR) * 255
    lsImg = lsImg.astype(np.uint8)
    # print(image)
    cv2.imwrite(output_img_path, lsImg)
    print(img2)
    cv2.imwrite(rgb_dir, img2)


dataset_dir = 'hls_in'
output_dir = 'hls_over_out'
rgb_output_dir = 'rgb_over_out'

# here do Adjustments

# Gets the image path to transform and generates the target path
image_filenames = [(os.path.join(dataset_dir, x), os.path.join(output_dir, x), os.path.join(rgb_output_dir, x))
                   for x in os.listdir(dataset_dir)]
# Convert all images
for path in image_filenames:
    update(path[0], path[1], path[2])
