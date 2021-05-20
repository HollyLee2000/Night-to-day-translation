import cv2
from matplotlib import pyplot as plt
import sys
import os

# the code for data augmentation
data_path = 'turn/'
out_path = 'turn_out/'
img_list = os.listdir(data_path)
if len(img_list) == 0:
    print('Data directory is empty.')
    sys.exit(0)

for img_name in img_list:
    #print(os.path.join(data_path, img_name));
    img = cv2.imread(os.path.join(data_path, img_name))
    name = img_name[:-4]
    xImg = cv2.flip(img, 1, dst=None)
    # xImg1 = cv2.flip(img,0,dst=None)
    # xImg2 = cv2.flip(img,-1,dst=None)
    cv2.imwrite(out_path + name + "_hm.jpg", xImg)

print("Done!")

# plt.show()
# xImg = cv2.flip(img,1,dst=None)
# xImg1 = cv2.flip(img,0,dst=None)
# xImg2 = cv2.flip(img,-1,dst=None)
# plt.subplot(221),plt.imshow(img)
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(222),plt.imshow(xImg)
# plt.title('Remap shuiping Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(223),plt.imshow(xImg1)
# plt.title('Remap chuizhi Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(224),plt.imshow(xImg2)
# plt.title('Remap duijiao Image'), plt.xticks([]), plt.yticks([])
#
# plt.show()
