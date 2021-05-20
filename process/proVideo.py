import cv2
import os
import subprocess
import imageio
import os
from PIL import Image
from moviepy.editor import *


def video2frame(videos_path, frames_save_path, time_interval):
    '''
    :param videos_path: 视频的存放路径
    :param frames_save_path: 视频切分成帧之后图片的保存路径
    :param time_interval: 保存间隔
    :return:
    '''
    vidcap = cv2.VideoCapture(videos_path)
    success, image = vidcap.read()
    count = 0
    while success:
        success, image = vidcap.read()
        count += 1
        if count % time_interval == 0:
            if image is not None:
                cv2.imencode('.jpg', image)[1].tofile(frames_save_path + "/frame%d.jpg" % count)
        # if count == 20:
        #   break
    print(count)


if __name__ == '__main__':
    data_path = 'video_in'  # 放待处理图片在这个目录下
    save_path = 'pic_out'  # 保存分帧的目录   要在最后面加一个标号，比如../../video1/
    music_path = 'music'  # 保存音频的目录   要在最后面加一个标号，比如../../music1
    video_list = os.listdir(data_path)
    i = 20
for video_name in video_list:
    i = i + 1
    videos_path = os.path.join(data_path, video_name)
    frames_save_path = save_path + '/'  # 在保存分帧的目录加后缀，如最终目录为../../video1/
    print(frames_save_path)
    time_interval = 2  # 隔一帧保存一次
    video2frame(videos_path, frames_save_path, time_interval)
    video = VideoFileClip(videos_path)
    audio = video.audio
    audio.write_audiofile(music_path + '/' + video_name[:len(video_name) - 4] + '.mp3')

# videos_path = 'D:/assignment/BigCreate/procVideo/originVideo/IMG_1930.mp4'
# frames_save_path = 'D:/assignment/BigCreate/procVideo/cutMusic/music1/'
# print(videos_path.split('/')[-1]);
