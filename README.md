# unparied night-to-day translation
## Introduction
This code is for night-to-day translation. Image processing codes is in the directory "process". In addition to 
the paper, we made some attempts at image sharpening and handling local over-exposure. The codes is developed by 
HollyLee, RuiZhu et al. And the work is based on many previous studies, To be specific:
<br>Cycle-GAN paper：https://arxiv.org/pdf/1703.10593.pdf, by YanJun Zhu et al.
<br>pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf, by Phillip Isola et al.
<br>To improve training of wasserstein GANs: https://arxiv.org/abs/1704.00028, by Gulrajani et al.
<br>To construct resnet generator, we adapt torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
And their paper: https://arxiv.org/pdf/1512.03385.pdf.
<br>DCP paper: http://kaiminghe.com/publications/cvpr09.pdf
<br>We adapt the idea of self-attention map from Enlighten-GAN: https://arxiv.org/abs/1906.06972
## Run the code for unpaired mode (only to generate pictures)
Put your images from night domain in ```./process/in_img``` and run ```./process/run.py``` for enhancing.   

Images after enhancement will be stored in ```./process/out_img``` and dehazing results of those enhanced images are in ```./process/final_out_img```.  

To achieve stlye-transfer, put yor enhanced night images(not dehazed) in ```./datasets/enhanced2daylight/trainA``` and 
daylight images in ```./datasets/enhanced2daylight/trainB```
- Train the model:
```
python train.py --dataroot ./datasets/enhanced2daylight --name enhanced2daylight_cyclegan --model cycle_gan
```
- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.  

  To see more intermediate results, check out `./checkpoints/enhanced2daylight_cyclegan/web/index.html`.  

  To test the model, you're supposed to put your test images in ```./datasets/enhanced2daylight/testA``` for enhanced images and ```./datasets/enhanced2daylight/testB ``` for daylight images  

- Test the model for translating both sides(always not necessary):
```
python test.py --dataroot ./datasets/enhanced2daylight --name enhanced2daylight_cyclegan --model cycle_gan
```
- Test the model for translating from only one side(e.g. from enhanced to daylight):
```
python test.py --dataroot datasets/enhanced2daylight/testA --name enhanced2daylight_cyclegan --model test --no_dropout
```
- The test results will be saved here: `./results/enhanced2daylight_cyclegan/latest_test`.
After style transfer, do the haze removal if necessary. Put hazed images in ```./process/defog_in``` and run ```./process/run_defog.py```
The final result will be in ```./process/defog_out```

##supplementary specification
In directory```./process```, there are some codes for image processing:<br>
```batch_rename.py```: Batch rename image files in the folder.<br>
```run_clearify.py```: Batch clearify image files in the folder.<br>
```run.py```: Batch process image files by Retinex in the folder. (and we adopt MSRCP)<br>
```run_defog.py```: Batch defog image files in the folder.<br>
```HLS.py```: Batch adjust brightness and contrast of image files in the folder.<br>
```HLS_overexposure.py```: some attempts to solve over-exposure.<br>
```proVideo.py```: separates video frames and sounds.<br>
```make_video.py```: make video.<br>
```find_dif.py```: Restoration of active objects erased by overfitting.<br>
