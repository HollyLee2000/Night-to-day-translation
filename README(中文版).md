# 非配对日夜翻译
## 介绍
这个工程用于非配对的夜间翻译。图像处理代码在“process”目录中。除了在论文中所提到的，我们
还进行了一些图像锐化和处理局部过度曝光的尝试。这项工作基于前人的许多研究，具体来说： 
<br>Cycle-GAN 论文：https://arxiv.org/pdf/1703.10593.pdf, 作者是朱彦俊等人。
<br>pix2pix 论文: https://arxiv.org/pdf/1611.07004.pdf, 作者是Phillip Isola等人。
<br>为了提高GAN的训练结果，计算损失时参考了: https://arxiv.org/abs/1704.00028, 作者是Gulrajani等人。
<br>为了构建残差网络生成器，采用了Justin Johnson、李飞飞等人的实时风格转移与超分辨项目的torch代码(https://github.com/jcjohnson/fast-neural-style)，
他们的论文: https://arxiv.org/pdf/1512.03385.pdf。
<br>DCP论文: http://kaiminghe.com/publications/cvpr09.pdf。
<br>我们从Enlighten-GAN那里采取了自正则图的想法，他们的论文: https://arxiv.org/abs/1906.06972。
## Run the code for unpaired mode (only to generate pictures)
把夜间图像域放入 ```./process/in_img```并运行 ```./process/run.py```以增强图像。
增强后的图像将存于 ```./process/out_img```他们直接去雾的结果将会存储在 ```./process/final_out_img```
将增强后的夜间图像放入 ```./datasets/enhanced2daylight/trainA```并将日光图像放入 ```./datasets/enhanced2daylight/trainB```
- 使用以下指令训练模型:
```
python train.py --dataroot ./datasets/enhanced2daylight --name enhanced2daylight_cyclegan --model cycle_gan
```
- 使用以下指令查看训练中的验证集结果和损失函数曲线： `python -m visdom.server`，在此网址查看结果：http://localhost:8097。
查看更多中间结果，请访问`./checkpoints/enhanced2daylight_cyclegan/web/index.html`
测试模型时，将你要测试的图像(增强后的夜间)放入 ```./datasets/enhanced2daylight/testA```，并将日光图像放入```./datasets/enhanced2daylight/testB ```

- 测试双方到彼此的映射(经常是没必要的):
```
python test.py --dataroot ./datasets/enhanced2daylight --name enhanced2daylight_cyclegan --model cycle_gan
```
- 只测试一方到另一方的映射(比如增强后的夜间到日光):
```
python test.py --dataroot datasets/enhanced2daylight/testA --name enhanced2daylight_cyclegan --model test --no_dropout
```
- 测试结果将会保存在: `./results/enhanced2daylight_cyclegan/latest_test`
风格迁移之后, 如有需要则进行去雾。将雾化图像放进```./process/defog_in```并运行```./process/run_defog.py```
最后的结果将保存在```./process/defog_out```

##补充说明
在目录```./process```下, 有一些用于图像处理的代码:<br>
```batch_rename.py```: 批量重命名<br>
```run_clearify.py```: 批量清晰化(通过锐化卷积和双边滤波)<br>
```run.py```: 批量MSRCP增强<br>
```run_defog.py```: 批量去雾<br>
```HLS.py```: 通过HLS色彩空间更改亮度对比度<br>
```HLS_overexposure.py```: 尝试直接解决过度曝光<br>
```proVideo.py```: 视频分帧、分离出音乐副本<br>
```make_video.py```: 分帧图像处理后，生成视频<br>
```find_dif.py```: 还原由于过拟合被抹去的活动物体<br>