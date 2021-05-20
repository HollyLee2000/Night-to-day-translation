import cv2
import os
import numpy as np
from PIL import Image
import subprocess
from moviepy.editor import *
import imageio


def frame2video(im_dir, video_dir, fps):
    im_list = os.listdir(im_dir)
    im_list.sort(key=lambda x: int(x.replace("frame", "").split('.')[0]))  # 最好再看看图片顺序对不
    img = Image.open(os.path.join(im_dir, im_list[0]))
    img_size = img.size  # 获得图片分辨率，im_dir文件夹下的图片分辨率需要一致

    # fourcc = cv2.cv.CV_FOURCC('M','J','P','G') #opencv版本是2
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # opencv版本是3
    videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)
    # count = 1
    for i in im_list:
        im_name = os.path.join(im_dir + i)
        frame = cv2.imdecode(np.fromfile(im_name, dtype=np.uint8), -1)
        videoWriter.write(frame)
        # count+=1
        # if (count == 200):
        #     print(im_name)
        #     break
    videoWriter.release()
    print('finish')


def video_add_mp3(file_name, mp3_file, i):
    """
     视频添加音频
    :param file_name: 传入视频文件的路径
    :param mp3_file: 传入音频文件的路径
    :return:
    """

    outfile_name = file_name.split('.')[0] + '-txt.mp4'
    subprocess.call('ffmpeg -i ' + file_name
                    + ' -i ' + mp3_file + ' -strict -2 -f avi '
                    + outfile_name, shell=True)


if __name__ == '__main__':
    fps = 14  # 帧率
    for i in range(1):
        # 这一部分是将分帧复原为视频
        data_path = 'tool2'  # 分帧的路径../../resultImgx      这里的x是从1到21   按照实际情况修改一下
        video_to = 'video_result'  # 恢复的视频存放的路径,可以按照实际情况修改一下

        im_dir = data_path + "/"  # 在分帧的路径后面加字符i和"/"
        video_dir = video_to + "dire_transfer" + ".mp4"  # video_to后面指的是合成视频的文件名，你可以根据实际情况修改

        frame2video(im_dir, video_dir, fps)  # 这一部分是将分帧复原为视频

        # 这一部分是把音频添加到视频里面
        music_path = 'music'
        music_dir = music_path + "/night2.mp3"
        outfile_name = video_to + "/dire_transfer" + ".mp4"
        #
        video = VideoFileClip(video_dir)
        audio = AudioFileClip(music_dir)
        videoclip2 = video.set_audio(audio)
        name1 = outfile_name
        videoclip2.write_videofile(name1)
        # 这一部分是把音频添加到视频里面

        print("The " + str(i) + "video is Done!")
