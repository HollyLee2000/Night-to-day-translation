import numpy as np
import cv2
import os

# 这段代码用于还原被抹去的活动物体
input_img_path = 'cankao.jpg'  # 参考的夜间无人图
img = cv2.imread(input_img_path)  # 参考的夜间无人图
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
count = 2
# file_handle = open('1.txt', mode='w')
# cv2.imwrite('video_final_out/img_gray.jpg', img_gray)
while count <= 646:
    img_night = cv2.imread('video_pic_in/frame' + str(count) + '.jpg')  # 夜间原视频帧
    night_gray = cv2.cvtColor(img_night, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite('video_final_out/img_night.jpg', night_gray)
    img_day = cv2.imread('try_in/frame' + str(count) + '.png')  # 已还原的白天图
    for i in range(560, img[:, :, 1].shape[0]):
        for j in range(300, img[:, :, 1].shape[1]):
            if abs(int(img_gray[i][j]) - int(night_gray[i][j])) > 30:
                img_day[i][j][0] = img_night[i][j][0]
                img_day[i][j][1] = img_night[i][j][1]
                img_day[i][j][2] = img_night[i][j][2]
    '''
    for i in range(img[:, :, 1].shape[0]):
        for j in range(img[:, :, 1].shape[1]):
            file_handle.write(str(img_gray[i][j]) + ' ')
        file_handle.write('\n')
    file_handle.write('好了\n')
    file_handle.write('好了\n')
    file_handle.write('好了\n')
    file_handle.write('好了\n')

    for i in range(img[:, :, 1].shape[0]):
        for j in range(img[:, :, 1].shape[1]):
            file_handle.write(str(night_gray[i][j]) + ' ')
        file_handle.write('\n')
    file_handle.write('\n')
    '''
    cv2.imwrite('video_final_out/frame' + str(count) + '.png', img_day)
    count = count + 2
