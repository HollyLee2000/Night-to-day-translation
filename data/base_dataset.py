"""
这个模块为数据集实现了一个抽象的基类BaseDataset ,也包括了一些通用的转换函数
"""
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod


class BaseDataset(data.Dataset, ABC):
    """
    这是一个抽象的基类，其子类需要实现以下四个函数：
    -- <__init__>:                      初始化该类
    -- <__len__>:                       返回数据集的大小
    -- <__getitem__>:                   获取一个随机的数据点
    -- <modify_commandline_options>:    额外选项，添加特定于数据集的命令参数，并设置默认选项
    """

    def __init__(self, opt):
        """
        初始化类，保存类中的选项，opt是BaseOptions的一个子类
        """
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """
        添加新的特定于数据集的命令参数，并重写现有参数的默认值
        参数:
            parser          -- 命令参数
            is_train (bool) -- 是属于训练的还是测试的

        返回值:
            修改后的命令参数
        """
        return parser

    @abstractmethod
    def __len__(self):
        """返回数据集的大小"""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """
        返回一个随机数据点及其元数据信息
        参数:
            index - - 用来建立数据索引的随机整数
        返回值:
            一个带有其名字的数据字典，包含数据本身及其元数据信息
        """
        pass


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':  # resize将原图像调整为正方形
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':  # scale设置宽大小并将图像按原比例调整
        new_w = opt.load_size
        new_h = opt.load_size * h // w  # //表示整数除法，向下取整，/是浮点除法

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))  # randint表示取两者间的随机整数
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5  # 图像随机进行左右镜像翻转，作为一种data augmentation的方法

    return {'crop_pos': (x, y), 'flip': flip}  # crop_pos表示随机裁剪的位置


#  BICUBIC：双三次插值算法
def get_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in opt.preprocess:  # 调整大小为正方形
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.preprocess:  # 调用__scale_width，按原比例调整大小
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, opt.crop_size, method)))

    if 'crop' in opt.preprocess:  # 随机裁剪
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    if opt.preprocess == 'none':  # 若无预处理选项，只需保证长和宽为4的倍数，若不满足则进行调整
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())  # 若无param,添加随机镜像翻转
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if convert:
        transform_list += [transforms.ToTensor()]  # 图像转tensor(C,H,W),对应通道数和尺寸
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]  # 标准化，各通道均值和标准差变为0.5
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __make_power_2(img, base, method=Image.BICUBIC):  # 保证图像长和宽为4的倍数，若不满足则进行调整
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_size, crop_size, method=Image.BICUBIC):   # 按原比例调整大小
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)


def __crop(img, pos, size):  # 随机裁剪，作为一种data augmentation的方法，裁剪范围是正方形
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if ow > tw or oh > th:
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    """
        if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)  # 如有需要旋转再取消注释
    """
    return img


def __print_size_warning(ow, oh, w, h):
    """打印尺寸警告，只出现一次，并自动调整"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
