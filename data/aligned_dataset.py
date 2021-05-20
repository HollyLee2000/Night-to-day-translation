import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image


class AlignedDataset(BaseDataset):
    """
    成对图像数据集的数据集类
    """
    def __init__(self, opt):
        """
        初始化类，保存类中的选项，opt是BaseOptions的一个子类
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))
        assert(self.opt.load_size >= self.opt.crop_size)
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

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
        # 读取给定一个随机整数索引的图像
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        # 将ab图像分成a和b
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # 对A和B同时应用相同的转换
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """返回数据集中中的图像总数"""
        return len(self.AB_paths)
