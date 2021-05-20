import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random


class UnalignedDataset(BaseDataset):
    """
    此数据集类可以加载未配对的数据集
    其中训练图像放在trainA、trainB下，测试图像放在testA、testB下
    """
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """
            返回一个随机数据点及其元数据信息
            参数:
                index - - 用来建立数据索引的随机整数
            返回值:
                返回一个包含a、b、a_paths和b_paths的字典
                A (tensor) - - 域中的一个图像
                B (tensor) - - 目标域中的对应图像
                A_paths (str) - - A图像路径
                B_paths (str) - - B图像路径
        """
        A_path = self.A_paths[index % self.A_size]  # %防止out of range
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:   # 随机化域b的索引，避免使用固定配对
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """
        返回数据集中中的图像总数
        此模式下两个图像的数量很可能不同，我们采取其中较大的
        """
        return max(self.A_size, self.B_size)
